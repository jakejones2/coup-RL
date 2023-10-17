# Coup Environment

Trying to train a keras model to play the card game **[Coup](https://www.ultraboardgames.com/coup/game-rules.php)**. This is a work in progress!

Environment nearly ready - missing a few subtleties of play such as the correct reinburcement when blocking a steal of 1 coin. Also need to redesign to take a variable number of players from 2-6 (this should be quite simple).

Bigger challenge is the RLlib algorithms. Currently unable to use the action mask in training, which is massively inefficient. Once this is fixed, need to optimise model layers and PPO settings. Then create a web app to play the game!

To see current progress, run `random_policy.py`. This plays the game with random decisions by each agent.

## Links (ignore):

https://docs.ray.io/en/master/rllib/rllib-sample-collection.html
https://discuss.ray.io/t/problem-with-action-masking/6145/8
https://docs.ray.io/en/latest/rllib/rllib-rlmodule.html

## Notes (ignore):

Old env observation space (precise bounds for each data point):

```
"observations": spaces.Tuple((
  Discrete(300), # turn
  Discrete(100), # player0 coins
  Discrete(100), # player1 coins
  Discrete(100), # player2 coins
  Discrete(100), # player3 coins
  Discrete(6), # card 1
  Discrete(6), # card 2
  Discrete(6), # card 3
  Discrete(6), # card 4
  Discrete(26), # player0 last move
  Discrete(26), # player0 second last move
  Discrete(26), # player0 third last move
  Discrete(26), # player1 last move
  Discrete(26), # player1 second last move
  Discrete(26), # player1 third last move
  Discrete(26), # player2 last move
  Discrete(26), # player2 second last move
  Discrete(26), # player2 third last move
  Discrete(26), # player3 last move
  Discrete(26), # player3 second last move
  Discrete(26), # player3 third last move
)),
```
