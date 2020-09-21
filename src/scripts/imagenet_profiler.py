import sys
import tensorflow as tf
from research.slim.nets.resnet_v1 import resnet_v1_50 
from research.slim.nets.resnet_v1 import resnet_v1_152
from research.slim.nets.vgg import vgg_16
from research.slim.nets.inception_v4 import inception_v4
import networkx as nx
import graph_profile_pb2


def get_synth_data(batch_size, height, width, num_channels, num_classes, dtype):
    with tf.device('cpu:0'):
        # Synthetic input should be within [0, 255].
        inputs = tf.random.truncated_normal([height, width, num_channels],
                                            dtype=dtype,
                                            mean=127,
                                            stddev=60,
                                            name='synthetic_inputs')
        labels = tf.random.uniform([1],
                                   minval=0,
                                   maxval=num_classes - 1,
                                   dtype=tf.int32,
                                   name='synthetic_labels')
        # Cast to float32 for Keras model.
        labels = tf.cast(labels, dtype=tf.int32)
        data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()
        # `drop_remainder` will make dataset produce outputs with known shapes.
        data = data.batch(batch_size, drop_remainder=True)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        iterator = data.make_initializable_iterator()
        next_element = iterator.get_next()
        it_init = iterator.initializer
    return next_element, it_init


def create_model(arch, batch_size, dtype, optimizer, num_classes=1001):
    with tf.device('gpu:0'):
        dtype = tf.float16 if dtype == 'float32' else float16
        if arch == 'resnet_v1_50':
            model = resnet_v1_50
            next_element, it_init = get_synth_data(batch_size, 224, 224, 3, num_classes, dtype)
            _, output_dict = model(next_element[0], num_classes, is_training=True)
            logits = output_dict['resnet_v1_50/logits']
            logits = tf.reduce_mean(logits, axis=[1, 2])

        elif arch == 'resnet_v1_152':
            model = resnet_v1_152
            next_element, it_init = get_synth_data(batch_size, 224, 224, 3, num_classes, dtype)
            _, output_dict = model(next_element[0], num_classes, is_training=True)
            logits = output_dict['resnet_v1_152/logits']
            logits = tf.reduce_mean(logits, axis=[1, 2])

        elif arch == 'vgg_16':
            model = vgg_16
            next_element, it_init = get_synth_data(batch_size, 224, 224, 3, num_classes, dtype)
            net, output_dict = model(next_element[0], num_classes, is_training=True)
            logits = net

        elif arch == 'inception_v4':
            next_element, it_init = get_synth_data(batch_size, 299, 299, 3, num_classes, dtype)
            logits, _ = inception_v4(next_element[0], num_classes)

        else:
            raise error_name('Unknown model.')

        loss = tf.losses.sparse_softmax_cross_entropy(labels=next_element[1], logits=logits)
        if optimizer == 'GD':
            train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
        elif optimizer == 'Adam':
            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

        init = [tf.initializers.global_variables(), it_init]
    return train_op, init


def filter_ops(graph_def):
    op_types = set()
    types_map = {}
    for op in tf.get_default_graph().get_operations():
        op_types.add(op.type)
        types_map[op.name] = op.type

    print('All op types: ', sorted(op_types))
    special_ops = [
        'BiasAdd', 'BiasAddGrad',
        'Conv2D', 'Conv2DBackpropFilter', 'Conv2DBackpropInput',
        'MaxPool', 'MaxPoolGrad',
        'Mean',
        'Relu', 'ReluGrad',
        'Softmax', 'SparseSoftmaxCrossEntropyWithLogits',
        'Tile',
        'VariableV2',
        'AddV2',
        'ApplyGradientDescent',
        'AddN', 'Identity', 'NoOp'
    ]
    print('We are keeping:', sorted(special_ops))

    g = nx.DiGraph()
    nodes_to_remove = set()
    for n in graph_def.node:
        if (n.name not in types_map) or (types_map[n.name] not in special_ops):
            nodes_to_remove.add(n.name)
        g.add_node(n.name)
        for pre in n.input:
            if pre == 'gradients/resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/Conv2D_grad/Conv2DBackpropInput':
                print('dadadadadadadadad', n)
            if (pre not in types_map) or (types_map[pre] not in special_ops):
                nodes_to_remove.add(pre)
            g.add_edge(pre, n.name)
    for n in nodes_to_remove:
        pred = g.predecessors(n)
        succ = g.successors(n)
        new_edges = [(p, s) for p in pred for s in succ]
        g.remove_node(n)
        g.add_edges_from(new_edges)

    for n in g.nodes:
        g.nodes[n]['op_type'] = types_map[n]
    return g


def run_model(train_op, init, warmup_runs=10, profile_runs=10):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    run_metadata = tf.RunMetadata()
    with tf.Session(config=config) as sess:
        sess.run(init)
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        for i in range(warmup_runs):
            sess.run(train_op)

        for i in range(profile_runs):
            sess.run(train_op,
                     options=options,
                     run_metadata=run_metadata)

    return sess.graph_def, run_metadata


def recurse_prof_proto(node, indent, graph):
    # print('--' * indent + node.name + ': %s %s' % (node.total_exec_micros, node.total_parameters) )
    indent = indent + 1
    if node.name in list(graph.nodes):
        graph.nodes[node.name]['op'] = node

    for i in range(len(node.children)):
        ch = tf.profiler.GraphNodeProto()
        ch.CopyFrom(node.children.pop())
        recurse_prof_proto(ch, indent, graph)


def generate_profile(graph, run_metadata):
    # Todo: check for children calls to have time estimations
    ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
    opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()
                                ).with_node_names(show_name_regexes=['.*']) \
        .select(['params', 'float_ops', 'micros', 'bytes']) \
        .with_max_depth(100000) \
        .with_min_execution_time() \
        .with_min_memory() \
        .with_step(-1) \
        .with_empty_output() \
        .build()

    prof = tf.profiler.profile(
        tf.get_default_graph(),
        run_meta=run_metadata,
        cmd='scope',
        options=opts)

    recurse_prof_proto(prof, 0, graph)

    gp = create_graph_profile(graph)

    return gp


def add_memread(profile, name, num_bytes):
    node = profile.nodes.add()
    node.name = name
    node.op_type = graph_profile_pb2.Profile.Op.OpType.MEMORY
    node.mem_type = graph_profile_pb2.Profile.Op.MemType.READVARIABLE
    node.num_bytes = num_bytes
    return profile


def add_memwrite(profile, name, num_bytes):
    node = profile.nodes.add()
    node.name = name
    node.op_type = graph_profile_pb2.Profile.Op.OpType.MEMORY
    node.mem_type = graph_profile_pb2.Profile.Op.MemType.WRITEVARIABLE
    node.num_bytes = num_bytes
    return profile


def add_comp(profile, name, comp_time_us, output_bytes):
    node = profile.nodes.add()
    node.name = name
    node.op_type = graph_profile_pb2.Profile.Op.OpType.COMPUTE
    node.comp_time_us = comp_time_us
    node.output_bytes = output_bytes
    return profile


def add_cntrl_dep(profile, name):
    node = profile.nodes.add()
    node.name = name
    node.op_type = graph_profile_pb2.Profile.Op.OpType.CONTROLDEPENDENCY
    return profile


def add_adj(profile, name, succs):
    adj = profile.graph.add()
    adj.node = name
    for s in succs:
        adj.succs.append(s)
    return profile


def create_graph_profile(graph):
    zero_deg_nodes = []
    for n in graph.nodes:
        if graph.degree(n) == 0:
            zero_deg_nodes.append(n)
    for n in zero_deg_nodes:
        graph.remove_node(n)
        print('removed degree=0: ', n)
    for n in graph.nodes:
        if graph.out_degree(n) == 0:
            print('out_degree=0', n)
    # sanity checks:
    try:
        cyc = nx.algorithms.cycles.find_cycle(graph)
        print('The graph has a cycle!', cyc)
        exit()
    except:
        pass

    nx.readwrite.adjlist.write_adjlist(graph, 'graph.nx')
    profile = graph_profile_pb2.Profile()
    for n in graph.nodes:
        op_type = graph.nodes[n]['op_type']
        op = graph.nodes[n]['op']

        if op_type == 'VariableV2':
            add_memread(profile, op.name, op.total_output_bytes)
        elif op_type == 'ApplyGradientDescent':
            preds = list(graph.predecessors(n))
            assert len(preds) == 2
            pred_types = [graph.nodes[p]['op_type'] for p in preds]
            assert 'VariableV2' in pred_types
            for p in preds:
                if graph.nodes[p]['op_type'] == 'VariableV2':
                    num_bytes = graph.nodes[p]['op'].total_output_bytes
            add_memwrite(profile, op.name, num_bytes)
        elif op_type in ['NoOp', 'Identity']:
            add_cntrl_dep(profile, op.name)
        else:
            add_comp(profile, op.name, op.accelerator_exec_micros, op.output_bytes)

        add_adj(profile, op.name, list(graph.successors(n)))

    return profile


def write_graph_profile(profile, filename):
    f = open(filename, "wb")
    f.write(profile.SerializeToString())
    f.close()


if __name__ == '__main__':
    batch_size = int(sys.argv[1])
    arch = sys.argv[2]
    train_op, init = create_model(arch=arch, batch_size=batch_size, dtype='float32', optimizer='GD',
                                  num_classes=1001)
    graph_def, run_metadata = run_model(train_op, init, warmup_runs=100, profile_runs=100)
    filtered_graph = filter_ops(graph_def)
    graph_profile = generate_profile(filtered_graph, run_metadata)
    write_graph_profile(graph_profile, '%s_float32_bs%d.pb' % (arch, batch_size))
