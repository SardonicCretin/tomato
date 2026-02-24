0. Explanation of what to expect from this (time, etc.)

1. About the interviewee
How many years of experience do you have growing paddy?
How big is your field/scale of operation
What varieties do you work with (have you worked with)?

2. Diagnosing process
How much of your time is dedicated to visual inspections (disease, pests + disease exclusively)
Do you primarily rely on: Visual inspection Laboratory confirmation Both?
At what growth stages do disease diagnoses most commonly occur?
When you visually diagnose a disease, what features do you look for first?
Which visual cues are most reliable? Color changes Lesion shape Texture Distribution pattern Location on plant (leaf, stem, neck, panicle, etc.)
Are there features that are frequently misleading?
Which diseases are easiest to confuse visually?
Do you rely on contextual clues (weather, season, location)?
Are early-stage symptoms visually distinct or ambiguous?
How often does correct diagnosis require non-visual information?

3. Variability & Edge Cases
How much visual variation exists within the same disease?
What causes that variation? Cultivar differences Growth stage Lighting Severity
Are symptoms consistent across leaves of the same plant?
How common are mixed infections (secondary infections)?
Can one disease manifest differently on: Leaves vs stems? Early vs late stage?
Are there “borderline” cases that even experienced farmers disagree on?
Which diseases are commonly overdiagnosed or underdiagnosed?
How to treat RARE DISEASES? More meaningful to detect rare or common diseases?

3. Image Acquisition & Data Quality
What makes an image diagnostically useful?
What makes an image unusable?
How important is focus, angle, and lighting?
Are close-up images more useful than full-plant images?
Is background context helpful or distracting?
Can disease be identified from a single image, or is multiple-view inspection needed?
Are smartphone images sufficient for diagnosis?
Do experts mentally “crop” images when inspecting them?

4. Labeling & Annotation Challenges
How confident are you when assigning a single disease label?
Are some images better labeled as “uncertain”?
Would multi-label annotations ever be appropriate?
Are severity labels (mild / moderate / severe) meaningful?
How often are ground-truth labels revised after further inspection?
Is it acceptable to label based on dominant disease only?
Which diseases require lab confirmation to be 100% certain?

5. Dataset Bias & Distribution Shift
Field vs controlled environments? Different regions or climates?
Are certain diseases more likely to be photographed?
Are rare diseases underrepresented because they are hard to capture?
Does disease prevalence change seasonally (rainy etc)?

6. Model Errors & Acceptability
Which errors are more harmful: False positives? False negatives?
Are there diseases that should never be missed?
Is it acceptable for a system to say “I don’t know”?
Would ranked predictions be more useful than a single label?
How much accuracy loss is acceptable in exchange for reliability?
Should a model prioritize early detection or confident detection?

7. Interpretability & Trust
Would visual explanations increase your trust in a system?
What kind of explanation would be most useful? Highlighted lesion areas Textual descriptions Comparison to known examples
Are there visual patterns a model might exploit that humans would reject?
Would incorrect explanations reduce trust even if predictions are correct?
How should a system communicate uncertainty?

8. System Use
How proficient with a phone camera


