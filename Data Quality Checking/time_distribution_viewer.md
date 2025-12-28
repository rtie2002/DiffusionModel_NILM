# Time Distribution Viewer - Mathematical Theory

## 📚 Overview

This document explains the mathematical theory behind sin/cos encoding for cyclical time features used in the Time Distribution Viewer.

---

## 🎯 Problem Statement

### Why Not Use Raw Time Values?

**Problem**: Time is cyclical, but raw numerical values don't capture this:

```
Raw values:
- 23:00 = 23
- 00:00 = 0
- Distance: |23 - 0| = 23 (very far!)

Reality:
- 23:00 and 00:00 are only 1 hour apart (very close!)
```

**Neural networks** trained on raw values would treat 23 and 0 as distant values, failing to learn the cyclical pattern.

---

## 🔵 Solution: Unit Circle Mapping

### Concept

Map time values to points on a **unit circle** (radius = 1):

```
           12:00/00:00 (0°)
              (0, 1)
                 |
                 |
  09:00 (-1,0) --+-- 03:00 (1, 0)
                 |
                 |
              (0,-1)
           18:00 (180°)
```

Each time value corresponds to a unique point `(cos θ, sin θ)` on the circle.

---

## 📐 Mathematical Formulation

### Encoding: Time → Sin/Cos

For a time value `t` with period `T`:

```
θ = 2π × (t / T)

x = cos(θ)  # x-coordinate
y = sin(θ)  # y-coordinate
```

**Where**:
- `t` ∈ [0, T-1] (e.g., minute: 0-59, hour: 0-23)
- `T` = period (60 for minutes, 24 for hours, 7 for days, 12 for months)
- `θ` = angle in radians [0, 2π]
- `(x, y)` = point on unit circle

**Properties**:
1. `x² + y² = 1` (always on unit circle)
2. Cyclical: `θ = 0` and `θ = 2π` map to same point
3. Distance on circle ≈ time difference

---

### Decoding: Sin/Cos → Time

Given `(sin_value, cos_value)`, recover time `t`:

```
Step 1: θ = arctan2(sin_value, cos_value)
        Returns angle in [-π, π]

Step 2: if θ < 0:
            θ = θ + 2π
        Convert to [0, 2π]

Step 3: t = (θ / 2π) × T
        Convert angle back to time

Step 4: t = round(t) mod T
        Round and ensure valid range
```

**Key Function: arctan2(y, x)**

Unlike regular `arctan(y/x)` which only returns [-π/2, π/2], `arctan2(y, x)` considers the signs of both arguments and returns the full range [-π, π], giving the correct quadrant.

```
Quadrant I   (x>0, y>0): arctan2(y, x) ∈ (0, π/2)
Quadrant II  (x<0, y>0): arctan2(y, x) ∈ (π/2, π)
Quadrant III (x<0, y<0): arctan2(y, x) ∈ (-π, -π/2)
Quadrant IV  (x>0, y<0): arctan2(y, x) ∈ (-π/2, 0)
```

---

## 🔢 Detailed Examples

### Example 1: Minute Encoding (Period = 60)

**Encoding**:

```python
# Minute 0 (start of hour)
t = 0
θ = 2π × 0/60 = 0°
sin = sin(0°) = 0.0
cos = cos(0°) = 1.0
Point: (1.0, 0.0) - rightmost point on circle

# Minute 15 (quarter hour)
t = 15
θ = 2π × 15/60 = π/2 = 90°
sin = sin(90°) = 1.0
cos = cos(90°) = 0.0
Point: (0.0, 1.0) - top of circle

# Minute 30 (half hour)
t = 30
θ = 2π × 30/60 = π = 180°
sin = sin(180°) = 0.0
cos = cos(180°) = -1.0
Point: (-1.0, 0.0) - leftmost point

# Minute 45 (three-quarter hour)
t = 45
θ = 2π × 45/60 = 3π/2 = 270°
sin = sin(270°) = -1.0
cos = cos(270°) = 0.0
Point: (0.0, -1.0) - bottom of circle

# Minute 59 (end of hour)
t = 59
θ = 2π × 59/60 ≈ 354°
sin ≈ -0.1045
cos ≈ 0.9945
Point: (0.9945, -0.1045) - almost back to start!
```

**Decoding**:

```python
# Given: sin = 0.0, cos = -1.0

Step 1: θ = arctan2(0.0, -1.0) = π

Step 2: π > 0, so no adjustment needed

Step 3: t = (π / 2π) × 60 = 0.5 × 60 = 30

Step 4: t = round(30) mod 60 = 30

Result: 30 minutes ✓
```

---

### Example 2: Hour Encoding (Period = 24)

**Encoding**:

```python
# Hour 0 (midnight)
θ = 2π × 0/24 = 0°
→ (cos=1.0, sin=0.0)

# Hour 6 (morning)
θ = 2π × 6/24 = π/2 = 90°
→ (cos=0.0, sin=1.0)

# Hour 12 (noon)
θ = 2π × 12/24 = π = 180°
→ (cos=-1.0, sin=0.0)

# Hour 18 (evening)
θ = 2π × 18/24 = 3π/2 = 270°
→ (cos=0.0, sin=-1.0)

# Hour 23 (late night)
θ = 2π × 23/24 ≈ 345°
→ (cos≈0.9659, sin≈-0.2588)
```

**Decoding Example**:

```python
# Given: sin = 1.0, cos = 0.0

θ = arctan2(1.0, 0.0) = π/2
t = (π/2 / 2π) × 24 = 6

Result: 6:00 ✓
```

---

### Example 3: Day of Week (Period = 7)

```python
# Monday (0)
θ = 2π × 0/7 = 0°
→ (cos=1.0, sin=0.0)

# Wednesday (2)
θ = 2π × 2/7 ≈ 102.86°
→ (cos≈-0.2225, sin≈0.9749)

# Sunday (6)
θ = 2π × 6/7 ≈ 308.57°
→ (cos≈0.6235, sin≈-0.7818)
```

---

## 🔄 Our Implementation

### Encoding (in `multivariate_ukdale_preprocess.py`)

```python
# Extract raw time values (0-based)
minute = df['time'].dt.minute      # 0-59
hour = df['time'].dt.hour          # 0-23
dayofweek = df['time'].dt.dayofweek  # 0-6 (0=Monday)
month = df['time'].dt.month        # 1-12

# Apply sin/cos encoding
df['minute_sin'] = np.sin(2 * np.pi * minute / 60.0)
df['minute_cos'] = np.cos(2 * np.pi * minute / 60.0)

df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

df['dow_sin'] = np.sin(2 * np.pi * dayofweek / 7.0)
df['dow_cos'] = np.cos(2 * np.pi * dayofweek / 7.0)

df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
df['month_cos'] = np.cos(2 * np.pi * month / 12.0)
```

### Decoding (in `time_distribution_viewer.py`)

```python
# Minute (convert to 1-60 for display)
angle = np.arctan2(df['minute_sin'], df['minute_cos'])
angle = np.where(angle < 0, angle + 2 * np.pi, angle)
df['minute'] = np.round((angle / (2 * np.pi)) * 60) % 60 + 1

# Hour (convert to 1-24 for display)
angle = np.arctan2(df['hour_sin'], df['hour_cos'])
angle = np.where(angle < 0, angle + 2 * np.pi, angle)
df['hour'] = np.round((angle / (2 * np.pi)) * 24) % 24 + 1

# Day of week (convert to 1-7 for display)
angle = np.arctan2(df['dow_sin'], df['dow_cos'])
angle = np.where(angle < 0, angle + 2 * np.pi, angle)
df['dow'] = np.round((angle / (2 * np.pi)) * 7) % 7 + 1

# Month (convert to 1-12 for display)
angle = np.arctan2(df['month_sin'], df['month_cos'])
angle = np.where(angle < 0, angle + 2 * np.pi, angle)
df['month'] = (np.round((angle / (2 * np.pi)) * 12) % 12) + 1
```

**Note**: The `+1` at the end converts from 0-based (0-59, 0-23, etc.) to 1-based (1-60, 1-24, etc.) for more intuitive display.

---

## ✅ Verification

### Mathematical Verification

For any valid sin/cos encoding:

```python
sin² + cos² = 1  # Always true for points on unit circle
```

This property can be used to verify correctness of encoded values.

### Round-trip Verification

```python
# Original
t_original = 30

# Encode
θ = 2π × 30/60 = π
sin_val = sin(π) = 0.0
cos_val = cos(π) = -1.0

# Decode
θ_back = arctan2(0.0, -1.0) = π
t_back = (π / 2π) × 60 = 30

# Verify
assert t_original == t_back  ✓
```

---

## 🎯 Advantages of Sin/Cos Encoding

### 1. Preserves Cyclical Nature

```
Distance on circle ≈ Actual time difference

Example (hours):
- 23:00 to 00:00: Arc length ≈ 1/24 of circle
- 12:00 to 13:00: Arc length ≈ 1/24 of circle
Both are correctly represented as similar distances!
```

### 2. Smooth Representation

```
Continuous mapping: small time changes → small coordinate changes
No discontinuities (unlike raw values where 23 → 0 is a jump)
```

### 3. Neural Network Friendly

```
- Two continuous features (sin, cos) instead of one discrete feature
- Smooth gradients for backpropagation
- Easier to learn cyclical patterns
```

### 4. Mathematically Elegant

```
- Well-defined inverse (arctan2)
- Verifiable (sin² + cos² = 1)
- No ambiguity (unique point for each time)
```

---

## 📊 Histogram Bin Alignment

To center histogram bars on their values:

```python
# For values 1-60
bins = np.arange(0.5, 61.5, 1)

# This creates bins:
# [0.5, 1.5] → center at 1.0
# [1.5, 2.5] → center at 2.0
# ...
# [59.5, 60.5] → center at 60.0
```

**Why 0.5 offset?**

Histogram bins are defined by edges. To center a bar at value `v`, we need edges at `v - 0.5` and `v + 0.5`.

---

## 🔬 Comparison with Raw Values

| Aspect | Raw Values | Sin/Cos Encoding |
|--------|-----------|------------------|
| Cyclical | ❌ No | ✅ Yes |
| Distance metric | ❌ Incorrect | ✅ Correct |
| Continuity | ❌ Discontinuous | ✅ Continuous |
| Dimensions | 1 | 2 |
| Range | [0, T-1] | [-1, 1] × [-1, 1] |
| ML-friendly | ❌ Poor | ✅ Good |

---

## 📚 References

1. **NILMFormer Paper**: Uses sin/cos encoding for time features
2. **Unit Circle**: Standard mathematical concept from trigonometry
3. **arctan2**: Standard function in NumPy and most math libraries

---

## 💡 Summary

**Sin/Cos encoding** transforms cyclical time features into a continuous, smooth representation on the unit circle, preserving the cyclical nature and making it easier for machine learning models to learn temporal patterns.

**Key Formula**:
```
Encode: (sin, cos) = (sin(2πt/T), cos(2πt/T))
Decode: t = (arctan2(sin, cos) / 2π) × T
```

**Verification**: sin² + cos² = 1 ✓
