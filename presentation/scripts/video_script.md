# BDAPP video script

Simple English, approximately 6–7 minutes with short pauses. Slide labels are cues, not spoken text. Slide 8 uses the measured Windows endpoint tolerance score; no assumed 87% result is presented as measured.

## Slide 1 — Introduction

Hello Professor. My name is Maral Farhatnejad.

My project started with a simple question: can electrical signals help us understand the condition of a brain organoid?

Brain organoids are small groups of brain cells grown in a lab. These cells can produce electrical signals.

I wanted to study whether those signals could tell us something about their age and how long they might keep working.

## Slide 2 — My goal for the final project

My goal for the final project is to build and test a system that uses neural activity to estimate culture age.

Ideally, I want to use a short recording, around one hour, to make that estimate.

I also want to study whether temperature and other environmental measurements help improve the prediction.

The wider goal is to explore remaining functional life: how long a culture may continue working.

To test this properly, I need clear age labels and reliable information about when function ends.

This project is my first step toward that goal.

## Slide 3 — Finding useful data

I worked with four data tracks.

The two FinalSpark datasets contained over ninety-four million events and about two-point-six million events.

I also studied DANDI recordings from rat cultures and Zenodo recordings from human organoids.

These helped me explore activity patterns, but they did not provide the individual age labels I needed.

A separate dataset from King's College London provided nine hundred and twenty recordings with clear culture-time labels.

Those labels measured days after cells were separated from organoids.

I kept the datasets separate because their materials and labels were different.

## Slide 4 — Building the pipeline

The first technical challenge was the amount of data.

I could not load all the event records into memory at once.

So, I built Python code that reads the data in smaller parts. It groups FinalSpark events into one-hour windows and calculates useful activity measures.

For the separate age benchmark, I used the published five-minute summaries.

I then compared a basic model with Ridge regression and a random forest.

The pipeline saved the predictions, scores, and checks so I could repeat the work and inspect the results.

## Slide 5 — Turning activity into numbers

The models needed numbers that describe the activity.

I used measures such as event counts, active channels, event size, and the time between events.

I also studied bursts, which are groups of events happening close together.

For FinalSpark, I examined environmental measurements and stimulation records where available.

The separate age benchmark used six published activity measures.

I kept age labels and recording names out of the model inputs. This helped make sure the model was learning from activity rather than identifying a recording.

## Slide 6 — Testing fairly

Next, I needed to test the models on data they had not seen before.

For FinalSpark, I trained on five hundred and nine hours from FS369 and tested on one hundred and fifteen hours from FS437.

For the age benchmark, I kept related recordings together and left whole plate and batch groups outside training.

I also tested donor cell lines that were left out of training.

This was important because repeated or closely related recordings could make the results look stronger than they really were.

## Slide 7 — FinalSpark model results

The shorter bars on this slide show lower average errors.

For calendar age, the count-only Ridge model reduced the error from about ninety-eight days to eighty-four days.

This was an improvement, but the age error was still large.

For time until the recorded experiment ended, the activity forest gave a clearer improvement.

It reduced the error from seven-point-four days to two-point-eight days. That is about sixty-two percent less error than the basic model.

However, these improvements did not yet show reliable prediction across experiments.

## Slide 8 — Looking more closely at the results

I also checked how often the endpoint predictions were close to the real recorded endpoint.

In the saved Windows test, eighty-four of the one hundred and fifteen hourly predictions were within two days.

That is about seventy-three percent.

About eighty-three percent were within three days.

These percentages describe this test experiment. They do not measure individual organoid lifespan.

The separate culture-age benchmark also showed a useful signal. Its forest model reduced average error from about eight days to six and a half days.

That was about nineteen percent less error than its basic model.

## Slide 9 — What I learned from the mistakes

The models still made important mistakes.

In the age benchmark, the youngest cultures were often predicted to be older. The oldest cultures were often predicted to be younger.

So, the predictions tended toward the middle ages.

I also learned that days after cells were separated is different from biological age.

For FinalSpark, missing events did not prove that the cells had stopped working, because recording coverage was not fully known.

These limits helped me understand what information a stronger model would need.

## Slide 10 — Conclusion and next steps

I started with a question about age and remaining function.

I finished this stage with a working Python pipeline and tested results from several data tracks.

Some models improved on the basic models, but important errors remained.

For the final project, I want to improve age prediction, test the value of environmental data, and evaluate the system on more independent cultures.

To study remaining functional life properly, I will also need clear individual follow-up and endpoint labels.

My goal is to build a system whose predictions are useful and supported by the data.

Thank you for listening.
