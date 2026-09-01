# Example clip attribution

The example clips bundled with this Space come from **ESC-50** (<https://github.com/karoldvl/ESC-50>), which curates clips from Freesound.

Every clip here was selected because its Freesound source is released under **CC0** (public domain dedication) — no attribution or non-commercial restriction attaches to them. They are credited anyway. Note that ESC-50 *as a dataset* is CC BY-NC; that applies to the collection, not to these individually-CC0 recordings.

Clips were converted to FLAC and downmixed to mono; otherwise unmodified.

| clip | character | Freesound source | author | license |
|---|---|---|---|---|
| `dog.flac` | transient — sharp onsets — where the model should be surprised | [Chihuahua Barks](http://www.freesound.org/people/Mewsel/sounds/208030/) | Mewsel | CC0 |
| `door_wood_knock.flac` | transient — sharp onsets — where the model should be surprised | [Window_Knocking_Door_Interior-perspective_Exterior_Bang_Aggressive.wav](http://www.freesound.org/people/tompallant/sounds/261068/) | tompallant | CC0 |
| `glass_breaking.flac` | transient — sharp onsets — where the model should be surprised | [Window breaking.MP3](http://www.freesound.org/people/m1a2t3z4/sounds/112213/) | m1a2t3z4 | CC0 |
| `church_bells.flac` | tonal — pitched, slowly evolving | [Bells of St Alkelda's Church, Giggleswick, England.flac](http://www.freesound.org/people/thaighaudio/sounds/125825/) | thaighaudio | CC0 |
| `clock_tick.flac` | rhythmic — regularly spaced events — easy for a linear predictor | [Old Scots Clock - Ticking.wav](http://www.freesound.org/people/fauxpress/sounds/42139/) | fauxpress | CC0 |
| `keyboard_typing.flac` | rhythmic — regularly spaced events — easy for a linear predictor | [Mechanical Keyboard Typing](http://www.freesound.org/people/Nmb910/sounds/234923/) | Nmb910 | CC0 |
| `rain.flac` | texture — stationary noise — easy for everyone | [rain.MP3](http://www.freesound.org/people/Mafon2/sounds/160999/) | Mafon2 | CC0 |
| `crackling_fire.flac` | texture — stationary noise — easy for everyone | [Fire.wav](http://www.freesound.org/people/Adam_N/sounds/164661/) | Adam_N | CC0 |

## Why this mix

The set deliberately spans the range that makes the demo's point. Transient clips (dog, door knock, glass breaking) produce visible surprise peaks. Rhythmic ones (clock tick, keyboard typing) are where a linear AR does best, so the model's deficit against it is starkest. Textures (rain, fire) are easy for every predictor and show how little the naive persistence baseline actually demands.

## The AR reference

`assets/ar_latent_fma.pt` is a linear AR(4) fitted in the model's own latent space on 400 FMA clips — the data the checkpoint was trained on. It ships pre-fitted so the reported skill is reproducible and does not depend on which clips happen to be bundled. Regenerate it with `LinearAR` + `tajepa.eval.fit_latent_ar`.
