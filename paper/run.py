import os
import time
from collections import namedtuple
from subprocess import Popen

RunConfig = namedtuple('RunConfig', ['model_name', 'step_size_sec', 'strategy', 'weights', 'num_iters', 'single'])
configs = [
    #          model_name       , step_size_sec , strategy      , weights   , num_iters , single
    RunConfig( 'resnet_v1_50'   , 1e-6          , "128:8:8192"  , 250e6     , 8192      , 39321 ),
    RunConfig( 'transformer'    , 1e-6          , "128:8:1024"  , 700e6     , 16384     , 11264 ),
    RunConfig( 'gpt2_355M'      , 1e-4          , "128:8:1024"  , 68e8      , 2403e3    , 2284  ),
]

def run_experiments():
    max_tokens = 5
    for cnfg in configs:    
        for topo_cmnd in ["run_ocs.sh", "run_ring.sh"]:#["run_elect.sh", "run_ocs.sh", "run_ring.sh"]:
            token_found = False
            while not token_found:
                time.sleep(5)
                screens_raw = os.popen('screen -ls').read().split('\n')
                screens = [x for x in screens_raw if len(x.split('.'))==2 and 
                                                     x.split('.')[1].startswith('sipml-')]
                used_tokens = len(screens)
                free_tokens = max_tokens - used_tokens
                if free_tokens > 0:
                    token_found = True
                    break
    
            cmnd = f'./{topo_cmnd} {cnfg.model_name}'
            if topo_cmnd == 'run_ocs.sh':
                cmnd += f' "--step_size_sec {cnfg.step_size_sec} --strategy {cnfg.strategy} --single_shot"'
            print(cmnd)
            proc = Popen(cmnd, shell=True)
            proc.wait()

def analyze_electricals():
    for cnfg in configs:
        model_name = cnfg.model_name
        strategy_template = 'logs/' + model_name + '/' + strat + '/elect/ng1024/np1024/bw%d/latency0/strategy.log'
        session_template =  'logs/' + model_name + '/' + strat + '/elect/ng1024/np1024/bw%d/latency0/session.log'
        tm_template =       'logs/' + model_name + '/' + strat + '/elect/ng1024/np1024/bw%d/latency0/tm_estimator.log'
        legends = []
        tta_vs_bw = []
        for bw in [128, 256, 512, 1024, 2048, 4096, 8192]:
            session_file = session_template % bw
            legends.append('BW = %d Gbps' % bw)
            print('Opening file %s' % session_file)
            with open(session_file, 'r') as f:
                hybrid_iter_time = int(f.read())
                weights = weights_dict[model_name]
                ring_time_fattree400 = (2. * weights * 8 / (min(bw, 400) * 1e9) / sim_step)
                ring_time_fattree200 = (2. * weights * 8 / (min(bw, 200) * 1e9) / sim_step)
                ring_time = (2. * weights * 8 / (bw * 1e9) / sim_step)
                sipml_hybrid_iter_time = hybrid_iter_time + ring_time
                fat200_hybrid_iter_time = hybrid_iter_time + ring_time_fattree200
                fat400_hybrid_iter_time = hybrid_iter_time + ring_time_fattree400

                print("warning: it's for DP only")
                sipml_dp_iter_time = dp_iter_time + ring_time
                fat400_dp_iter_time = dp_iter_time + ring_time_fattree400 
                fat200_dp_iter_time = dp_iter_time + ring_time_fattree200
                tta_vs_bw.append([bw, 
                                  sipml_hybrid_iter_time, 
                                  fat200_hybrid_iter_time, 
                                  fat400_hybrid_iter_time,
                                  sipml_dp_iter_time,
                                  fat200_dp_iter_time,
                                  fat400_dp_iter_time])

            tm_file = tm_template % bw
            print('Opening file %s' % tm_file)
            with open(tm_file, 'r') as f:
                tm = f.read()
                tm = tm.split('\n')
                tm = [[float(x) for x in y.split(' ')[:-1]] for y in tm[:-1]]
                tm = np.array(tm)
                print(tm.sum())
        #pl.xlabel('MP Degree')
        #pl.ylabel('Time to Accuracy (mins)')
        #pl.xscale('log', base=2)
        #pl.legend(legends)
        #pl.show()    
        
        tta_vs_bw = np.array(tta_vs_bw)
        for i in range(6):
            pl.plot(tta_vs_bw[:,0], tta_vs_bw[:,i+1] * iters_dict[model_name] * sim_step / 60., 'o-')
        leg = ['IdealElect-Hybrid', '200Gbps-Hybrid', '400Gbps-Hybrid', 'IdealElect-DP', '200Gbps-DP', '400Gbps-DP']
        for i in range(6):
            l = leg[i]
            dashed = 'dashed' if l.split('-')[1]=='DP' else ''
            c = {'IdealElect': 'C', '400Gbps': 'B', '200Gbps': 'A'}[l.split('-')[0]] 
            with open('%s/%s.tex' % (model_name, leg[i]), 'w') as f:
                f.write('\\addplot+ [thick, Set1-%s, %s] coordinates{' % (c, dashed))
                for n in range(tta_vs_bw[:,0].shape[0]):
                    f.write('(%f, %f)\n' % (tta_vs_bw[n,0], tta_vs_bw[n,i+1] * iters_dict[model_name] * sim_step / 60.))
                f.write('};')    

        for i in range(3):
            with open('%s/%s.tex' % (model_name, leg[i].split('-')[0]+'-ratio'), 'w') as f:
                f.write('\\addplot+ [thick] coordinates{')
                for n in range(tta_vs_bw[:,0].shape[0]):
                    f.write('(%f, %f)\n' % (tta_vs_bw[n,0], tta_vs_bw[n,i+4] / tta_vs_bw[n,i+1]))
                f.write('};')    
        #pl.plot(tta_vs_bw[:,0], tta_vs_bw[:,3] / tta_vs_bw[:,1], 'o-')
        #pl.plot(tta_vs_bw[:,0], tta_vs_bw[:,2])
        pl.xlabel("BW (Gbps)")
        pl.ylabel('Iteration Time (ms)')
        pl.xscale('log', base=2)
        #pl.legend(['Runtime Measurement', 'Placement Estimate'])
        return leg

def plot():
    analyze_electricals()

if __name__ == '__main__':
    run_experiments()
    #plot() 
