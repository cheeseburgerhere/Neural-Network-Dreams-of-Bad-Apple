# Architecture block for the article

```text
First 16 source frames -> frozen encoder -> normalized seed latents
                                                |
Past 16 latent grids + their differences --------+
                  |
       concatenate state and velocity channels
                  |
       temporal U-Net: 16 -> 8 -> 4 time steps
       skip connections: 4 -> 8 -> 16 time steps
                  |
         final-step spatial features
                  |
      slow velocity + masked fast velocity
                  |
        last latent + velocity ---------------------------+
                                                          |
Time -> Fourier features -> embedding -> U-Net            |
  +-> two nearest anchor times -> learned anchor blend ---+
                                                          |
                                          spatial/cut-gated correction
                                                          |
                                                 next predicted latent
                                                   +-> feed back into history
                                                   +-> denormalize -> frozen decoder
                                                                          |
                                                      learned time-only polarity
                                                                          |
                                                             threshold -> binary frame
```

Each latent is 64 x 24 x 32. The U-Net pools time, not the spatial latent dimensions. Anchor timestamps are fixed; their value tensors are learned. Correction uses a spatial gate, not a fixed scalar blend.

In normalized latent coordinates:

    motion = previous_latent + slow_velocity + masked_fast_velocity
    memory = sum(address_weight_i(time) * learned_anchor_i)
    next_latent = motion + gate * (memory - motion)

Memory-only replaces the recurrence with next_latent = memory, keeping the decoder and polarity restoration. It removes history, temporal prediction and fusion together.
