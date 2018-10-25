#-*- coding: utf-8 -*-
"""
    Bin-Packing using Neural Combinational Optimization.

    Author: Ruben Solozabal, PhD student at the University of the Basque Country [UPV-EHU] Bilbao
    Date: October 2018
"""
import logging
import tensorflow as tf
from environment import *
from service_batch_generator import *
from agent import *
from config import *
from solver import *
from tensorflow.python import debug as tf_debug
from tqdm import tqdm

""" Globals """
DEBUG = True

if __name__ == "__main__":

    """ Log """
    logging.basicConfig(level=logging.DEBUG)  # TODO: filename='example.log'
    # DEBUG, INFO, WARNING, ERROR, CRITICAL

    """ Configuration """
    config, _ = get_config()

    """ Environment """
    env = Environment(config.num_bins, config.num_slots, config.num_descriptors)

    """ Batch of Services """
    services = ServiceBatchGenerator(config.batch_size, config.min_length, config.max_length, config.num_descriptors)

    """ Agent """
    state_size_sequence = config.max_length
    state_size_embeddings = config.num_descriptors     #OH Vector embedding
    action_size = config.num_bins
    agent = Agent(state_size_embeddings, state_size_sequence, action_size, config.batch_size, config.learning_rate, config.hidden_dim, config.num_stacks)

    """ Configure Saver to save & restore model variables """
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)

    print("Starting session...")
    with tf.Session() as sess:

        # Activate Tensorflow CLI debugger
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Activate Tensorflow debugger in Tensorboard
        #sess = tf_debug.TensorBoardDebugWrapperSession(
        #    sess=sess,
        #    grpc_debug_server_addresses=['localhost:6064'],
        #    send_traceback_and_source_code=True)

        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Print total number of parameters
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print('Shape: ', shape, 'Variables: ', variable_parameters)
            total_parameters += variable_parameters
        print('Total_parameters: ', total_parameters)

        # Restore variables from disk
        if config.load_model:
            saver.restore(sess, "save/tf_binpacking.ckpt")
            print("Model restored.")

        # Train model
        if config.train_mode:

            # Summary writer
            writer = tf.summary.FileWriter("summary/repo", sess.graph)

            # Main Loop
            print("\n Starting training...")
            for e in tqdm(range(config.num_epoch)):

                # New batch of states
                services.getNewState()

                # Vector embedding
                input_state = vector_embedding(services)

                # Compute placement
                feed = {agent.input_: input_state, agent.input_len_: [item for item in services.serviceLength]}
                positions = sess.run(agent.ptr.positions, feed_dict=feed)

                reward = np.zeros(config.batch_size)

                # Compute environment
                for batch in range(config.batch_size):
                    env.clear()
                    env.step(positions[batch], services.state[batch], services.serviceLength[batch])
                    reward[batch] = env.reward

                # RL Learning
                feed = {agent.reward_holder: [item for item in reward], agent.positions_holder: positions,
                        agent.input_: input_state, agent.input_len_: [item for item in services.serviceLength]}

                summary, _ = sess.run([agent.merged, agent.train_step], feed_dict=feed)

                if e % 100 == 0:
                    print("\n Mean batch ", e, "reward:", np.mean(reward))
                    writer.add_summary(summary, e)

                # Save intermediary model variables
                if config.save_model and e % max(1, int(config.num_epoch / 5)) == 0 and e != 0:
                    save_path = saver.save(sess, "save/tmp.ckpt", global_step=e)
                    print("\n Model saved in file: %s" % save_path)

                e += 1

            print("\n Training COMPLETED!")

            if config.save_model:
                save_path = saver.save(sess, "save/tf_binpacking.ckpt")
                print("\n Model saved in file: %s" % save_path)

        # Test model
        else:
            # New batch of states
            services.getNewState()

            # Vector embedding
            input_state = vector_embedding(services)

            # Compute placement
            feed = {agent.input_: input_state, agent.input_len_: [item for item in services.serviceLength]}
            positions = sess.run(agent.ptr.positions, feed_dict=feed)

            reward = np.zeros(config.batch_size)

            # Compute environment
            for batch in range(config.batch_size):
                env.clear()
                env.step(positions[batch], services.state[batch], services.serviceLength[batch])
                reward[batch] = env.reward

                # Render some batch services
                if batch % max(1, int(config.batch_size / 5)) == 0:
                    print("\n Rendering batch ", batch, "...")
                    env.render(batch)

            # Calculate performance
            if config.enable_performance:

                print("\n Calculating optimal solutions... ")
                optReward = np.zeros(config.batch_size)

                for batch in tqdm(range(config.batch_size)):
                    optPlacement = solver(services.state[batch], services.serviceLength[batch], env)
                    env.clear()
                    env.step(optPlacement, services.state[batch], services.serviceLength[batch])
                    optReward[batch] = env.reward
                    assert optReward[batch] + 0.1 > reward[batch]  # Avoid inequalities in the last decimal...

                performance = np.sum(reward) / np.sum(optReward)
                print("\n Performance: ", performance)
