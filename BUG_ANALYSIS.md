# Bug Analysis: Observation Processing and Inference Logic

## Summary
Analysis of potential bugs in the transformer training pipeline's observation processing and inference logic.

## Files Analyzed
1. `CapyFormer/examples/train_transformer_debug.py` - Training script and evaluation
2. `CapyFormer/capyformer/hf_trainer.py` - Trainer and inference wrappers  
3. `CapyFormer/capyformer/data.py` - Dataset and normalization stats
4. `metamachine/environments/components/state.py` - Environment observation conversion

## Key Findings

### 1. ✅ Key Matching (VERIFIED CORRECT)
- **Dataset expects**: `module0`, `module1`, `module2`, `module3`, `module4` (line 39 of `train_transformer_debug.py`)
- **Environment provides**: `flat_obs_to_dict()` returns keys `module0`, `module1`, etc. (line 1875 of `state.py`)
- **Status**: ✅ Keys match correctly

### 2. ✅ Normalization Stats Flow (VERIFIED CORRECT)
- **Dataset**: Computes `input_mean` and `input_std` as dicts with token names as keys
- **Dataset**: Aliases to `state_mean` and `state_std` (line 650-651 of `data.py`)
- **Dataset**: Returns via `get_state_stats()` (line 820 of `data.py`)
- **Trainer**: Passes to model as `state_mean` and `state_std` (line 2031-2032 of `hf_trainer.py`)
- **Model**: Converts numpy arrays to torch tensors (line 924-925 of `hf_trainer.py`)
- **Inference**: Accesses via `model.state_mean` and `model.state_std` (line 1124-1125 of `hf_trainer.py`)
- **Status**: ✅ Flow is correct

### 3. ✅ Shape Handling (VERIFIED CORRECT)
**Location**: `ActionChunkingInference._normalize_state()` (line 1142-1149 of `hf_trainer.py`)

**Verification**:
- `flat_obs_to_dict()` returns 1D numpy arrays: `flat_obs[offset:offset + per_module_size]` creates 1D array (line 1875 of `state.py`)
- Normalization stats are computed as 1D arrays: `np.nanmean(states_concat, axis=0)` where `states_concat` is `(T, dim)`, result is `(dim,)` (line 646 of `data.py`)
- When normalizing: `value` is 1D tensor, `mean` and `std` are 1D tensors, broadcasting works correctly
- **Status**: ✅ Shapes are compatible

### 4. ⚠️ **POTENTIAL BUG: Missing Key Normalization**
**Location**: `ActionChunkingInference.step()` (line 1165-1182 of `hf_trainer.py`)

**Issue**: 
```python
if is_present:
    value = current_state[name]
    # ... convert to tensor ...
    value = self._normalize_state(name, value)  # ✅ Normalized
    self.mask_history[name].append(torch.tensor(True))
else:
    token_dim = self.input_token_dims[idx]
    value = torch.zeros(token_dim, dtype=torch.float32)
    # ❌ NOT normalized - zero tensor is unnormalized!
    self.mask_history[name].append(torch.tensor(False))
```

**Problem**: 
- Missing keys get unnormalized zero tensors
- During training, missing keys might be handled with NaN padding that gets masked out
- This inconsistency could cause issues if the model expects normalized inputs

**Impact**: Medium - Only affects cases where keys are missing, which shouldn't happen in normal operation but could cause silent failures.

**Recommendation**: 
```python
else:
    token_dim = self.input_token_dims[idx]
    value = torch.zeros(token_dim, dtype=torch.float32)
    # Normalize the zero tensor for consistency
    value = self._normalize_state(name, value)
    self.mask_history[name].append(torch.tensor(False))
```

### 5. ⚠️ **POTENTIAL BUG: Silent Normalization Failure**
**Location**: `ActionChunkingInference._normalize_state()` (line 1142-1149 of `hf_trainer.py`)

**Issue**:
```python
def _normalize_state(self, name: str, value: torch.Tensor) -> torch.Tensor:
    if self.state_mean is not None and self.state_std is not None:
        if name in self.state_mean and name in self.state_std:
            # Normalize
            return (value - mean) / std
    return value  # ❌ Silently returns unnormalized value
```

**Problem**:
- If a key is expected but missing from normalization stats, normalization silently fails
- No warning or error is raised
- This could lead to incorrect model inputs that are hard to debug

**Impact**: High - Could cause incorrect model behavior if keys don't match between dataset and inference.

**Recommendation**: Add validation and logging:
```python
def _normalize_state(self, name: str, value: torch.Tensor) -> torch.Tensor:
    if self.state_mean is not None and self.state_std is not None:
        if name in self.state_mean and name in self.state_std:
            mean = self.state_mean[name].to(value.device)
            std = self.state_std[name].to(value.device)
            return (value - mean) / std
        else:
            # Key expected but not found in stats
            import warnings
            warnings.warn(
                f"Normalization stats not found for key '{name}'. "
                f"Available keys: {list(self.state_mean.keys())}. "
                f"Returning unnormalized value."
            )
    elif name in self.input_token_names:
        # Stats should exist but don't
        import warnings
        warnings.warn(
            f"Normalization stats not initialized but key '{name}' is expected. "
            f"Returning unnormalized value."
        )
    return value
```

### 6. ✅ Device Handling (VERIFIED CORRECT)
- Normalization stats are moved to the correct device in `_normalize_state()` (line 1146-1147)
- **Status**: ✅ Correctly handles device placement

### 7. ✅ Observation Shape from Environment (VERIFIED CORRECT)
- `flat_obs_to_dict()` returns 1D numpy arrays for each module (line 1875 of `state.py`)
- Arrays are created via slicing: `flat_obs[offset:offset + per_module_size]`
- **Status**: ✅ Shapes are correct (1D arrays)

## Summary of Issues

### Critical Issues: None Found ✅

### Medium Priority Issues:
1. **Missing key normalization**: Missing keys get unnormalized zeros (Issue #4)
2. **Silent normalization failure**: No warning when normalization stats are missing (Issue #5)

### Low Priority / Code Quality:
- Add debug logging for normalization operations
- Add validation for shape consistency

## Recommended Fixes

### Fix 1: Normalize missing keys
**File**: `CapyFormer/capyformer/hf_trainer.py`  
**Location**: `ActionChunkingInference.step()` around line 1178

```python
else:
    token_dim = self.input_token_dims[idx]
    value = torch.zeros(token_dim, dtype=torch.float32)
    value = self._normalize_state(name, value)  # Add this line
    self.mask_history[name].append(torch.tensor(False))
```

### Fix 2: Add validation to normalization
**File**: `CapyFormer/capyformer/hf_trainer.py`  
**Location**: `ActionChunkingInference._normalize_state()` around line 1142

Add warning when normalization fails (see recommendation in Issue #5 above).

## Testing Checklist

- [x] Verify `flat_obs_to_dict()` returns correct keys and shapes ✅
- [x] Verify normalization stats contain all expected keys ✅  
- [ ] Test with missing keys in `current_state` (should normalize zeros)
- [x] Test normalization with different tensor shapes ✅
- [ ] Add unit tests for normalization edge cases
- [ ] Test end-to-end: training → inference with same data
