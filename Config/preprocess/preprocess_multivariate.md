# Preprocessing Configuration and Normalization

## Normalization Parameters
The `preprocess_multivariate.yaml` configuration file defines the following global normalization constants for the aggregate (mains) power readings:

```yaml
normalization:
  aggregate_mean: 522
  aggregate_std: 814
```

These specific values (522 and 814) are derived from the statistics of the UK-DALE dataset used in the NILM-main project.

## Mathematical Formula
The normalization process uses **Z-score normalization** (also known as standardization). This technique rescales the data distribution so that the mean of the observed values is 0 and the standard deviation is 1.

The formula is:

$$
z = \frac{x - \mu}{\sigma}
$$

Where:
*   $z$ is the normalized output value fed into the model.
*   $x$ is the raw power reading (in Watts).
*   $\mu$ is the population mean (`aggregate_mean` = 522).
*   $\sigma$ is the population standard deviation (`aggregate_std` = 814).

## Implementation
In the preprocessing scripts (`multivariate_ukdale_preprocess_*.py`), this formula is applied directly to the aggregate column:

```python
# Normalization step in code
df_align['aggregate'] = (df_align['aggregate'] - args.aggregate_mean) / args.aggregate_std
```

### Why use this?
1.  **Model Stability**: Neural networks converge faster and are more stable when inputs are centered around 0 with low variance, rather than raw power values which can range from 0 to 5000+ Watts.
2.  **Scale Invariance**: It ensures that the scale of the input data does not arbitrarily influence the magnitude of the gradients during backpropagation.
