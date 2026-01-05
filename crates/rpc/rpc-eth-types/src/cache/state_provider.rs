//! Cached state provider for RPC operations.
//!
//! This module provides a caching layer for historical state access during RPC operations
//! like `debug_traceTransaction`. It caches account, storage, and bytecode reads to avoid
//! repeated expensive database lookups when replaying transactions.

use alloy_primitives::{map::HashMap, Address, BlockNumber, StorageKey, StorageValue, B256};
use reth_errors::ProviderResult;
use reth_primitives_traits::{Account, Bytecode};
use reth_storage_api::{
    AccountReader, BlockHashReader, BytecodeReader, HashedPostStateProvider, StateProofProvider,
    StateProvider, StateRootProvider, StorageRootProvider,
};
use reth_trie::{
    updates::TrieUpdates, AccountProof, HashedPostState, HashedStorage, MultiProof,
    MultiProofTargets, StorageMultiProof, StorageProof, TrieInput,
};
use revm::database::BundleState;
use std::cell::RefCell;

/// A caching wrapper around a [`StateProvider`] that caches account, storage, and bytecode reads.
///
/// This is designed for RPC operations that need to access historical state multiple times,
/// such as transaction tracing where prior transactions need to be replayed. By caching reads,
/// we avoid repeated expensive database lookups to historical state tables.
///
/// # Example
///
/// ```ignore
/// let state_provider = provider.state_by_block_id(block_id)?;
/// let cached_provider = CachedStateProvider::new(state_provider);
/// // Use cached_provider for multiple state accesses
/// ```
#[derive(Debug)]
pub struct CachedStateProvider<S> {
    /// The underlying state provider
    inner: S,
    /// Account cache: `address` -> `Option<Account>`
    account_cache: RefCell<HashMap<Address, Option<Account>>>,
    /// Storage cache: `(address, key)` -> `Option<StorageValue>`
    storage_cache: RefCell<HashMap<(Address, StorageKey), Option<StorageValue>>>,
    /// Bytecode cache: `code_hash` -> `Option<Bytecode>`
    code_cache: RefCell<HashMap<B256, Option<Bytecode>>>,
}

impl<S> CachedStateProvider<S> {
    /// Creates a new [`CachedStateProvider`] wrapping the given state provider.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            account_cache: RefCell::new(HashMap::default()),
            storage_cache: RefCell::new(HashMap::default()),
            code_cache: RefCell::new(HashMap::default()),
        }
    }

    /// Returns a reference to the inner state provider.
    pub const fn inner(&self) -> &S {
        &self.inner
    }

    /// Consumes self and returns the inner state provider.
    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S: AccountReader> AccountReader for CachedStateProvider<S> {
    fn basic_account(&self, address: &Address) -> ProviderResult<Option<Account>> {
        // Check cache first
        if let Some(cached) = self.account_cache.borrow().get(address) {
            return Ok(*cached);
        }

        // Cache miss - fetch from underlying provider
        let result = self.inner.basic_account(address)?;
        self.account_cache.borrow_mut().insert(*address, result);
        Ok(result)
    }
}

impl<S: StateProvider> StateProvider for CachedStateProvider<S> {
    fn storage(
        &self,
        address: Address,
        storage_key: StorageKey,
    ) -> ProviderResult<Option<StorageValue>> {
        let key = (address, storage_key);

        // Check cache first
        if let Some(cached) = self.storage_cache.borrow().get(&key) {
            return Ok(*cached);
        }

        // Cache miss - fetch from underlying provider
        let result = self.inner.storage(address, storage_key)?;
        self.storage_cache.borrow_mut().insert(key, result);
        Ok(result)
    }

    fn account_code(&self, addr: &Address) -> ProviderResult<Option<Bytecode>> {
        self.inner.account_code(addr)
    }

    fn account_balance(&self, addr: &Address) -> ProviderResult<Option<alloy_primitives::U256>> {
        self.inner.account_balance(addr)
    }

    fn account_nonce(&self, addr: &Address) -> ProviderResult<Option<u64>> {
        self.inner.account_nonce(addr)
    }
}

impl<S: BytecodeReader> BytecodeReader for CachedStateProvider<S> {
    fn bytecode_by_hash(&self, code_hash: &B256) -> ProviderResult<Option<Bytecode>> {
        // Check cache first
        if let Some(cached) = self.code_cache.borrow().get(code_hash) {
            return Ok(cached.clone());
        }

        // Cache miss - fetch from underlying provider
        let result = self.inner.bytecode_by_hash(code_hash)?;
        self.code_cache.borrow_mut().insert(*code_hash, result.clone());
        Ok(result)
    }
}

impl<S: BlockHashReader> BlockHashReader for CachedStateProvider<S> {
    fn block_hash(&self, number: BlockNumber) -> ProviderResult<Option<B256>> {
        self.inner.block_hash(number)
    }

    fn canonical_hashes_range(
        &self,
        start: BlockNumber,
        end: BlockNumber,
    ) -> ProviderResult<Vec<B256>> {
        self.inner.canonical_hashes_range(start, end)
    }
}

impl<S: StateRootProvider> StateRootProvider for CachedStateProvider<S> {
    fn state_root(&self, hashed_state: HashedPostState) -> ProviderResult<B256> {
        self.inner.state_root(hashed_state)
    }

    fn state_root_from_nodes(&self, input: TrieInput) -> ProviderResult<B256> {
        self.inner.state_root_from_nodes(input)
    }

    fn state_root_with_updates(
        &self,
        hashed_state: HashedPostState,
    ) -> ProviderResult<(B256, TrieUpdates)> {
        self.inner.state_root_with_updates(hashed_state)
    }

    fn state_root_from_nodes_with_updates(
        &self,
        input: TrieInput,
    ) -> ProviderResult<(B256, TrieUpdates)> {
        self.inner.state_root_from_nodes_with_updates(input)
    }
}

impl<S: StateProofProvider> StateProofProvider for CachedStateProvider<S> {
    fn proof(
        &self,
        input: TrieInput,
        address: Address,
        slots: &[B256],
    ) -> ProviderResult<AccountProof> {
        self.inner.proof(input, address, slots)
    }

    fn multiproof(
        &self,
        input: TrieInput,
        targets: MultiProofTargets,
    ) -> ProviderResult<MultiProof> {
        self.inner.multiproof(input, targets)
    }

    fn witness(
        &self,
        input: TrieInput,
        target: HashedPostState,
    ) -> ProviderResult<Vec<alloy_primitives::Bytes>> {
        self.inner.witness(input, target)
    }
}

impl<S: StorageRootProvider> StorageRootProvider for CachedStateProvider<S> {
    fn storage_root(
        &self,
        address: Address,
        hashed_storage: HashedStorage,
    ) -> ProviderResult<B256> {
        self.inner.storage_root(address, hashed_storage)
    }

    fn storage_proof(
        &self,
        address: Address,
        slot: B256,
        hashed_storage: HashedStorage,
    ) -> ProviderResult<StorageProof> {
        self.inner.storage_proof(address, slot, hashed_storage)
    }

    fn storage_multiproof(
        &self,
        address: Address,
        slots: &[B256],
        hashed_storage: HashedStorage,
    ) -> ProviderResult<StorageMultiProof> {
        self.inner.storage_multiproof(address, slots, hashed_storage)
    }
}

impl<S: HashedPostStateProvider> HashedPostStateProvider for CachedStateProvider<S> {
    fn hashed_post_state(&self, bundle_state: &BundleState) -> HashedPostState {
        self.inner.hashed_post_state(bundle_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::U256;
    use reth_storage_api::noop::NoopProvider;

    #[test]
    fn test_cached_state_provider_creation() {
        let provider = NoopProvider::default();
        let cached = CachedStateProvider::new(provider);
        assert!(cached.account_cache.borrow().is_empty());
        assert!(cached.storage_cache.borrow().is_empty());
        assert!(cached.code_cache.borrow().is_empty());
    }
}

