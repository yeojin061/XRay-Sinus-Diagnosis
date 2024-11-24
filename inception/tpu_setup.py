import tensorflow as tf

def setup_tpu():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU")
    except ValueError:
        strategy = tf.distribute.get_strategy()
        print("Running on CPU/GPU")
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    return strategy
