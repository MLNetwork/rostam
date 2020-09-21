import sys
import numpy as np
import networkx as nx
import json

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import transformer
import graph_profile_pb2

import tensorflow.compat.v1 as tf
import tensorflow.contrib.slim as slim

INPUT_LENGTH = 38
TARGET_LENGTH = 36
VOCAB_SIZE = 8000

def get_model(hparams=None, mode=tf.estimator.ModeKeys.TRAIN,
              has_input=False, model_cls=transformer.Transformer,
              batch_size=64 ):

  hparams = transformer.transformer_tall()
  hparams.use_tpu = False
  if hparams.get("problem_hparams", None) is None:
    p_hparams = problem_hparams.test_problem_hparams(VOCAB_SIZE,
                                                     VOCAB_SIZE,
                                                     hparams)
  if not has_input:
    del p_hparams.modality["inputs"]
  hparams.problem_hparams = p_hparams

  with tf.device('cpu:0'):
    inputs = np.random.randint(
      VOCAB_SIZE, size=(batch_size, INPUT_LENGTH, 1, 1))
    targets = np.random.randint(
      VOCAB_SIZE, size=(batch_size, TARGET_LENGTH, 1, 1))
    features = {
      "targets": tf.constant(targets, dtype=tf.int32, name="targets"),
      "target_space_id": tf.constant(1, dtype=tf.int32)
  }
    if has_input:
      features["inputs"] = tf.constant(inputs, dtype=tf.int32, name="inputs")

  return model_cls(hparams, mode, p_hparams), features

def create_model( arch, batch_size, dtype, optimizer ):
  with tf.device('gpu:0'):
    model, features = get_model(transformer.transformer_big(), batch_size=batch_size)
    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    if optimizer == 'GD':
        train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
    elif optimizer == 'Adam':
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    init = tf.initializers.global_variables()
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('total num. of parameters: %d' % total_parameters)
  print(tf.trainable_variables())
  return train_op, init

def filter_ops(graph_def):
    op_types = set()
    types_map = {}
    for op in tf.get_default_graph().get_operations():
        op_types.add(op.type)
        types_map[op.name] = op.type

    print('All op types: ', sorted(op_types))
    special_ops = [
        'AddN', 'AddV2', 
        'BatchMatMulV2',
        'BiasAdd', 'BiasAddGrad',
        'ConcatV2',
        'Cos',
        'Exp',
        'GatherV2'
        'Log', 'MatMul', 'Max', 'Maximum', 'Mean', 'Mul', 'Neg',
        'ResourceApplyGradientDescent',
        'Relu', 'ReluGrad',
        'Rsqrt', 'RsqrtGrad',
        'Sin',
        'Softmax', 'SoftmaxCrossEntropyWithLogits', 'SparseSoftmaxCrossEntropyWithLogits',
        'Transpose',
        'convert_gradient_to_tensor_HBc3xYw22Mw',
        'VarHandleOp',
        'ReadVariableOp',
        'NoOp', 'Identity',
    ]
    print('We are keeping:', sorted(special_ops))
    for name in types_map:
        if types_map[name] in special_ops:
            print(name)
    g = nx.DiGraph()
    nodes_to_remove = set()
    for n in graph_def.node:
        if (n.name not in types_map) or (types_map[n.name] not in special_ops):
            nodes_to_remove.add(n.name)
        g.add_node(n.name)
        for pre in n.input:
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
    config.allow_soft_placement = True
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
    #print('--' * indent + node.name + ': %s %s' % (node.total_exec_micros, node.total_parameters) )
    indent = indent + 1
    if node.name in list(graph.nodes):
        graph.nodes[node.name]['op'] = node

    for i in range(len(node.children)):
        ch = tf.profiler.GraphNodeProto()
        ch.CopyFrom(node.children.pop())
        recurse_prof_proto(ch, indent, graph)


def generate_profile(graph, run_metadata, read_varsize_map):
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

    gp = create_graph_profile(graph, read_varsize_map)

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


def create_graph_profile(graph, read_varsize_map):
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
    
    largest_size = 0
    for sg in nx.weakly_connected_components(graph):
        if len(list(sg)) > largest_size:
            largest_size = len(list(sg))
            largest_sg = list(sg)
    print('pre subprune size:', len(graph.nodes))
    to_remove = []
    for n in graph.nodes:
        if not n in largest_sg:
            to_remove.append(n)
    for n in to_remove:
        graph.remove_node(n)
    print('after subprune size:', len(graph.nodes))
    # sanity checks:
    try:
        cyc = nx.algorithms.cycles.find_cycle(graph)
        print('The graph has a cycle!', cyc)
        exit()
    except:
        pass

    nx.readwrite.adjlist.write_adjlist(graph, 'graph.nx')
    profile = graph_profile_pb2.Profile()
    total_mem_read = 0
    for n in graph.nodes:
        op_type = graph.nodes[n]['op_type']
        op = graph.nodes[n]['op']

        if op_type == 'ReadVariableOp':
            num_bytes = read_varsize_map[op.name]
            add_memread(profile, op.name, num_bytes)
            total_mem_read += num_bytes
            print('total_mem_read=', total_mem_read)
        elif op_type == 'ResourceApplyGradientDescent':
            preds = list(graph.predecessors(n))
            assert len(preds) == 2
            pred_types = [graph.nodes[p]['op_type'] for p in preds]
            assert 'VarHandleOp' in pred_types
            found = 0
            for p in preds:
                if graph.nodes[p]['op_type'] == 'VarHandleOp':
                    num_bytes = graph.nodes[p]['op'].total_output_bytes
                    found += 1
            assert found == 1
            add_memwrite(profile, op.name, num_bytes)
        elif op_type in ['NoOp', 'Identity', 'VarHandleOp']:
            add_cntrl_dep(profile, op.name)
        else:
            add_comp(profile, op.name, op.accelerator_exec_micros, op.output_bytes)

        add_adj(profile, op.name, list(graph.successors(n)))

    return profile


def write_graph_profile(profile, filename):
    f = open(filename, "wb")
    f.write(profile.SerializeToString())
    f.close()

def get_var_size(variable):
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    return variable_parameters

if __name__ == '__main__':
    batch_size = int(sys.argv[1])
    train_op, init = create_model(arch='transformer_big', batch_size=batch_size, dtype='float32', optimizer='GD' )
    graph_def, run_metadata = run_model(train_op, init, warmup_runs=100, profile_runs=100)
    var_ops = set()
    var_size_map = {}
    for n in graph_def.node:
        if n.op == "ReadVariableOp":
            assert( len(n.input) == 1 )
            var_ops.add(n.input[0])
    matched = set()
    all_g_vars = set()
    for v in tf.global_variables( ):
        g_var = v.name.split(':')[0]
        var_size_map[g_var] = get_var_size(v)
        all_g_vars.add(g_var)
        if g_var in var_ops:
            matched.add(g_var)
           
    print('found:', matched)
    print('var_ops - matched:', var_ops - matched)
    print('all_g_vars - matched:', all_g_vars - matched)
    assert( len(matched) == len(var_ops) )

    read_varsize_map = {}
    for n in graph_def.node:
        if n.op == "ReadVariableOp":
            assert( len(n.input) == 1 )
            read_varsize_map[n.name] = var_size_map[n.input[0]] * 4 

    filtered_graph = filter_ops(graph_def)
    graph_profile = generate_profile(filtered_graph, run_metadata, read_varsize_map)
    write_graph_profile(graph_profile, 'transformer_big_float32_hs_384_bs%d.pb' % batch_size)
