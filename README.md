# Sketch Recognition from Scratch

> A convolutional neural network that recognizes free-hand sketches across all 345 categories of
> Google's [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset), trained from scratch in PyTorch.

**Free-hand sketch recognition** is image classification for drawings made by people, not cameras:
a few quick strokes, no texture or color, and every person draws a "cat" differently. This model
is the vision half of [Sell Anything](https://github.com/alina0607/sell-anything), a game where the
player draws something with the mouse and a room of AI buyers bids on it.

> 🚧 **Status:** this repository currently holds the description only. The code is being moved out of
> the Sell Anything monorepo, with its tests and the teaching notebook, and will land here next.

## Results

On the held-out test set (172,500 drawings never seen in training, 500 per class):

| Metric | Score |
|---|---|
| Top-1 accuracy | **71.7%** |
| Top-3 accuracy | **87.1%** |

The game shows the top 3 guesses and lets the player pick, so top-3 is the number that matters for
play. For a sense of the hard cases: on 2,000 unseen *cat* drawings the model is right first time
68.7% of the time (85.2% in the top 3), and its usual mistakes are raccoon, dog, rabbit, cow, pig
and tiger — most Quick, Draw! cats are faces, so a full-body cat on four legs looks a lot like a cow.

## How it works

### 1. Strokes become the dataset's own bitmaps

A model is only as good as the match between what it trained on and what it sees. Player strokes
are rasterized the way Quick, Draw! made its 28×28 bitmaps: scaled to fit 0–255, resampled at 1-pixel
spacing and simplified with Ramer–Douglas–Peucker (ε = 2), then centered in a 256×256 box with 16-pixel
round strokes and 16 pixels of padding, and anti-aliased down to 28×28.
Rendering the dataset's raw strokes this way reproduces the official bitmaps with a **pixel
correlation of 0.996**.

### 2. The network

A VGG-style CNN, 1,912,985 parameters:

| Stage | Layers | Output |
|---|---|---|
| Block 1 | 2 × (3×3 conv, 64) + BatchNorm + ReLU, max-pool | 64 × 14 × 14 |
| Block 2 | 2 × (3×3 conv, 128) + BatchNorm + ReLU, max-pool | 128 × 7 × 7 |
| Block 3 | 3×3 conv, 256 + BatchNorm + ReLU, max-pool | 256 × 3 × 3 |
| Head | dropout 0.3 → linear 2304 → 512 + BatchNorm + ReLU → dropout 0.3 → linear 512 → 345 | class scores |

### 3. Training

- **Data:** 6,000 drawings per class, split 5,000 / 500 / 500 into training, validation and test
  (1.73 million training drawings), downloaded with HTTP range requests so only the needed slice of
  each class file is fetched.
- **Augmentation:** a random rotation (±12°), shift (±8%) and scale (0.9–1.1) per drawing, on the GPU.
- **Optimizer:** SGD with momentum 0.9 and a one-cycle learning-rate schedule (peak 0.2), 15 epochs,
  about 1.5 hours on an Apple M-series GPU. Training resumes from checkpoints.

### 4. A notebook that gets there one step at a time

The teaching notebook starts from a random guess and lowers the loss one idea at a time — linear
model, MLP, CNN, BatchNorm, dropout and augmentation, learning-rate schedule — measuring what each
step buys.

## What's next

- More drawings per class (Quick, Draw! has over 100,000 per class; this model uses 6,000), the most
  direct way to cover drawing styles like the full-body cat.
- Mirroring the drawing at test time was measured and does not help (cat top-1 68.7% → 69.3%, within noise).

## Data and license

Quick, Draw! by Google Creative Lab, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
