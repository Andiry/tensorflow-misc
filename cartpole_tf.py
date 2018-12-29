import tensorflow as tf
from tensorflow.contrib.layers import fully_connected as fc
import gym
import numpy as np

env = gym.make("CartPole-v0")

n_inputs = 4
n_hidden = 4
n_outputs = 1
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = fc(X, n_hidden, activation_fn=tf.nn.elu,
            weights_initializer=initializer)
logits = fc(hidden, n_outputs, activation_fn=None,
            weights_initializer=initializer)
outputs = tf.nn.sigmoid(logits)

plr = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(plr), num_samples=1)

y = 1. - tf.to_float(action)
cross_enthopy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                        logits=logits)
optimizer = tf.train.AdamOptimizer(0.01)
grads_vars = optimizer.compute_gradients(cross_enthopy)
gradients = [grad for grad, variable in grads_vars]
gradient_placeholders = []
grads_vars_feed = []

for grad, variable in grads_vars:
  gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
  gradient_placeholders.append(gradient_placeholder)
  grads_vars_feed.append((gradient_placeholder, variable))
  pass

training_op = optimizer.apply_gradients(grads_vars_feed)

init = tf.global_variables_initializer()
#saver = tf.train.Saver()

def discount_rewards(rewards, discount_rate):
  discount_rewards = np.empty(len(rewards))
  cumulative_rewards = 0
  for step in reversed(range(len(rewards))):
    cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
    discount_rewards[step] = cumulative_rewards
    pass
  return discount_rewards

def discount_normalize_rewards(all_rewards, discount_rate):
  all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                            for rewards in all_rewards]
  flat_rewards = np.concatenate(all_discounted_rewards)
  reward_mean = flat_rewards.mean()
  reward_std = flat_rewards.std()
  return [(discounted_rewards - reward_mean) / reward_std
          for discounted_rewards in all_discounted_rewards]

n_iterations = 50
n_max_steps = 1000
n_games_per_update = 10
save_iterations = 10
discount_rate = 0.95

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

with tf.Session() as sess:
  init.run()
  for iter in range(n_iterations):
    print('Iteration', iter)
    all_rewards = []
    all_gradients = []
    for game in range(n_games_per_update):
      current_rewards = []
      current_gradients = []
      obs = env.reset()
      for step in range(n_max_steps):
        action_val, gradients_val = sess.run([action, gradients],
                                             feed_dict = {X: obs.reshape(1, n_inputs)},
                                             options=options, run_metadata=run_metadata)
        obs, reward, done, info = env.step(action_val[0][0])
        current_rewards.append(reward)
        current_gradients.append(gradients_val)
        if done:
          break
      all_rewards.append(current_rewards)
      all_gradients.append(current_gradients)

    all_rewards = discount_normalize_rewards(all_rewards, discount_rate)
    feed_dict = {}
    for var_index, grad_placeholder in enumerate(gradient_placeholders):
      mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                for game_index, rewards in enumerate(all_rewards)
                                for step, reward in enumerate(rewards)], axis=0)
      feed_dict[grad_placeholder] = mean_gradients
    sess.run(training_op, feed_dict=feed_dict,
             options=options, run_metadata=run_metadata)
#    if iter % save_iterations == 0:
#      saver.save(sess, "/tmp/cartpole.ckpt")

  tf.train.write_graph(sess.graph, '/tmp/', 'cartpole.pbtxt')
  open('/tmp/cartpole-stepstats.pbtxt', 'w').write(str(run_metadata.step_stats))
  writer = tf.summary.FileWriter('/tmp/cartpole', sess.graph)
  writer.close()

  meta_graph_def = tf.train.export_meta_graph(filename='/tmp/cartpole.meta')
