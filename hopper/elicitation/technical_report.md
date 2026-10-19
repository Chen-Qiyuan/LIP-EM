# Hopper Simulation Environment

The Hopper is a planar one-legged robot simulated in MuJoCo with
three actuated revolute joints (thigh, leg, foot) connecting a
torso to a foot through a thigh and leg segment. The agent applies
a normalized torque (in `[-1, +1]`) to each joint at every
time-step.

## Data format

Each CSV is a replay buffer of state-action transitions
(~1M transitions per file) collected during RL training.

## CSV columns

`episode, step, z, pitch, q_thigh, q_leg, q_foot, x_dot, z_dot, pitch_dot,
qd_thigh, qd_leg, qd_foot, a_thigh, a_leg, a_foot, terminal`

| symbol     | meaning                                       | units    |
|------------|-----------------------------------------------|----------|
| z          | torso height above ground                     | m        |
| pitch      | torso pitch angle                             | rad      |
| q_*        | joint angle (thigh / leg / foot)              | rad      |
| x_dot      | torso forward velocity                        | m/s      |
| z_dot      | torso vertical velocity (positive = upward)   | m/s      |
| pitch_dot  | torso angular velocity                        | rad/s    |
| qd_*       | joint angular velocity                        | rad/s    |
| a_*        | normalized joint torque, in [-1, +1]          | -        |
| terminal   | 1 if last step of episode (Hopper fell)       | bool     |

Time-step `Δt = 0.008 s`. Rows within an episode are consecutive in
time; the state on row k+1 is the result of applying row k's
action.
