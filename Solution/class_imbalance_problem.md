# Class Imbalance in Diffusion Models for NILM

## 1. The Problem Observation
When we trained the Diffusion Model using the **Full Real Dataset** (instead of just pre-filtered "ON" windows), we observed a critical failure mode:
- **Input:** The model was given the entire history of appliance usage (e.g., months of data).
- **Output:** The generated samples contained **significantly fewer "ON" periods** than the real data. The model effectively learned to predict "OFF" (0 Watts) almost all the time.
- **Result:** The model failed to capture the true characteristics of the appliance, functioning mostly as a "silence generator."

## 2. The Mathematical Cause: "The Lazy Student"
This is a classic machine learning issue known as **Class Imbalance**, compounded by the **L1 Loss Function**.

### The Data Distribution
In a real household, appliances like a washing machine are **sparse**:
- **99% of the time:** The machine is OFF (Value $\approx$ 0).
- **1% of the time:** The machine is ON (Value $>0$).

### The Standard Training Objective (L1 Loss)
The model minimizes the **Mean Absolute Error (L1)**:

$$ L = \frac{1}{N} \sum_{i=1}^{N} | y_{predicted} - y_{true} | $$

Where:
- $y_{predicted}$ is the model's guess.
- $y_{true}$ is the real power value.

### Why the Model "Gave Up"
Let's look at the math from the model's perspective. It wants to minimize its total error score.

**Scenario A: The Model tries to learn the ON cycle.**
- It guesses "ON" sometimes.
- Because "ON" events are rare and complex (complex shapes, specific times), it often guesses wrong (timing mismatch, amplitude mismatch).
- **Result:** High Error penalty.

**Scenario B: The "Lazy" Strategy (Predicting 0).**
- The model simply guesses **0 (OFF)** for every single timestamp.
- **For 99% of the data (OFF periods):** The guess is perfect. Error = 0.
- **For 1% of the data (ON periods):** The guess is wrong. Error = $|0 - Power_{ON}|$.
- **Total Error Score:** Very low.

Mathematically, **L1 Loss optimizes for the Median of the data**. In a dataset where 99% of values are 0, the median is **0**. Therefore, the mathematically "optimal" solution for the model—to verify it understands the assignment—is to predict 0 everywhere.

The model didn't "fail"; it succeeded in finding the path of least resistance.

## 3. The Solution: Weighted Loss Function
To fix this, we changed the "scoring rules" of the game. We cannot change the dataset (the reality is that appliances are rarely used), so we must change the **Cost Function**.

We implemented a **Weighted Loss** that penalizes errors during "ON" periods much more heavily than mistakes during "OFF" periods.

### The New Mathematical Objective
$$ L_{weighted} = \frac{1}{N} \sum_{i=1}^{N} w_i \cdot | y_{predicted} - y_{true} | $$

Where $w_i$ is the weight for sample $i$:
- If $y_{true}$ is **OFF** ($< -0.9$): $w_i = 1.0$
- If $y_{true}$ is **ON** ($> -0.9$): $w_i = 20.0$ (Assigned 20x importance)

### The Logic Shift
Now, let's re-evaluate the model's strategies:

**Scenario B (The Lazy Strategy):**
- The model guesses 0 everywhere.
- **For 99% of data:** Error is 0.
- **For 1% of data (ON periods):** The error is calculated, **AND THEN MULTIPLIED BY 20**.
- **Result:** A massive spike in Loss. The "Lazy Strategy" is no longer viable.

**Scenario A (Learning the ON cycle):**
- To reduce that massive $20\times$ penalty, the model is forced to attempt to generate values $>0$ whenever it sees even a faint signal that an event might be occurring.
- It learns that "Missing an ON event" is 20 times worse than "Hallucinating an ON event when there isn't one."

### Outcome on Generation
By shifting the risk/reward ratio:
1.  **Training:** The model aggressively learns the features of the ON state because they are now "high value" data points.
2.  **Sampling:** When generating data, the model is less inhibited. If the noise suggests a potential event, the model amplifies it into a full ON cycle (because that's what minimized loss during training), rather than suppressing it to 0.

This results in a generated dataset that statistically matches the **frequency** of ON periods in the real world, rather than just matching the **median** value (which is zero).
