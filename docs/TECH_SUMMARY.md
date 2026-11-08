# Sip: a home care robot for drinks and tablets

*AgentVerse 2026, robotics track. Simulated Hello Robot Stretch 3 in MuJoCo. Code: github.com/Ibad-10/Sip*

**User and problem.** An older person living at home who can speak but has weak or shaky arms. Drinking and taking tablets on time both go wrong without help. About 4.5 million people aged 50+ in England need help with everyday tasks (Age UK, 2025). Medication-related harm affects about 1 in 3 older adults after hospital discharge and costs the NHS about £400m a year. A carer cannot be there every hour.

**What it does.** On "I'd like some water", Sip:
1. finds the glass with its cameras
2. pours from the jug and adds a straw
3. finds her
4. brings the straw to her lips

A safety gate controls every centimetre of that last move. At dose time (set in the home profile), Sip:
1. fetches the pill organiser and checks it is today's compartment
2. watches her take the tablets (hand at mouth, compartment empty)
3. waits for her "taken", logs it and keeps a streak

A mismatch, a decline, no answer, a suspected fall or a software fault raises a carer alert with a photo, and the alert repeats until a carer acknowledges it. In a new home Sip explores from an empty map and names the rooms from what it sees.

**Nothing is pre-set.** The robot works out:
- objects and the person: from the head camera
- the mouth: from face landmarks
- its own position: from wheel encoders, the IMU gyro and lidar scan matching (1–3 cm in a normal run)
- whether a grasp worked: from the wrist depth camera
- the map: from lidar and depth while exploring

Benchmarks use **randomised homes** (objects, seat and start pose change every run). Results: [fill in: success rate, sim labels vs real models, time per drink].

**Architecture.** Perception → beliefs (object memory, places she's been seen, map) → task state machine → safety gate → arm/base control. A trained arm policy is used only if it passed ≥90% on held-out tests; otherwise the classic controller runs. The dashboard shows each step and which model or rule decided it.

| Model | Job | Runs on | Latency |
|---|---|---|---|
| YOLO-World v2 (open vocabulary) | find objects, person, furniture | robot CPU, background thread | ~200 ms |
| CLIP + our few-shot classifier | glass vs mug vs jug vs straws | robot CPU | ~30 ms; **89% on unseen homes** |
| MediaPipe Face / Pose | mouth, head turn, hand at mouth, raised hand | robot CPU | ~20 ms |
| PPO policies (Stable-Baselines3) | reach-above-glass (100% held-out); grasp 62%, straw-to-lips 68% (still training; not used yet) | trained on Modal, run on robot | <1 ms |
| faster-whisper base.en | speech to text | robot CPU | ~1 s |
| Gemini 4 Argon (optional) | second opinion when unsure, new-home vocabulary, pill-box label, carer summary | cloud, capped at 30 calls | 2–10 s |
| Gemma 4 / ElevenLabs (optional) | small talk / voice | cloud; offline fallbacks | 1–10 s |

Everything safety-critical runs on the robot; it works with the network cut.

**Critical failure mode: a straw pushed into the face.** The safety gate:
- slows to 3 cm/s inside 30 cm
- only goes the last stretch after she says "ready"
- backs off if she turns her head (more than 30°), raises a hand or the face is lost
- stops at once on "stop" or "don't go" (negation-aware)
- pauses if perception is more than 1 s old
- retreats on contact force over 10 N

A judge-style test proves the contact check can fail (a straw pushed into the face must register).

**Limits.**
- The grip is a weld once the wrist camera sees the object between the fingers, and the water is visual: MuJoCo has no fluids.
- The straw snaps into the glass, and the glass is turned by setting its pose.
- Her motion is keyframed (Mixamo, retargeted), not physics.
- One person per home; multi-person is on the roadmap.
- Pill-box day reading needs Gemini (otherwise the clock is used).
- Learned straw/offer skills are not yet at 90%.

**Next.** Several people with face profiles; GPU training with MuJoCo Playground (Warp); a real Stretch 3 in a care setting with a carer present.
