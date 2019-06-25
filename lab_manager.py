from brian2 import *


def connect_network(conn_params, states, net, train = True):
    G_AL, G_KC, G_GGN, G_BL = states

    g_ALKC = conn_params['ex_ALKC']
    ALKC_synapses = conn_params['synALKC_class'](g_ALKC)
    S_ALKC = Synapses(G_AL, G_KC,
                      model = ALKC_synapses.eqs(),
                      on_pre = ALKC_synapses.onpre(),
                      on_post = ALKC_synapses.onpost(),
                      namespace = ALKC_synapses.namespace(),
                      method = ALKC_synapses.method())


    if train:
        S_ALKC.connect(p = conn_params['PALKC'])
    else:
        S_ALKC_conn = conn_params['S_ALKC_conn']
        S_ALKC.connect(i = S_ALKC_conn['i'], j = S_ALKC_conn['j'])

    S_ALKC.set_states(ALKC_synapses.init_cond())

    g_KCGGN = conn_params['ex_KCGGN']
    KCGGN_synapses = conn_params['synKCGGN_class'](g_KCGGN)
    S_KCGGN = Synapses(G_KC, G_GGN,
                      model = KCGGN_synapses.eqs(),
                      on_pre = KCGGN_synapses.onpre(),
                      on_post = KCGGN_synapses.onpost(),
                      namespace = KCGGN_synapses.namespace(),
                      method = KCGGN_synapses.method())

    S_KCGGN.connect()
    S_KCGGN.set_states(KCGGN_synapses.init_cond())

    g_GGNKC = conn_params['in_GGNKC']
    GGNKC_synapses = conn_params['synGGNKC_class'](g_GGNKC)
    S_GGNKC = Synapses(G_GGN, G_KC,
                      model = GGNKC_synapses.eqs(),
                      on_pre = GGNKC_synapses.onpre(),
                      on_post = GGNKC_synapses.onpost(),
                      namespace = GGNKC_synapses.namespace(),
                      method = GGNKC_synapses.method())

    S_GGNKC.connect()
    S_GGNKC.set_states(GGNKC_synapses.init_cond())

    g_KCBL = conn_params['ex_KCBL']
    eta = conn_params['etaSTDP']
    taupre = conn_params['taupreSTDP']
    taupost = conn_params['taupostSTDP']
    KCBL_synapses = conn_params['synKCBL_class'](g_KCBL, eta, taupre, taupost)
    S_KCBL = Synapses(G_KC, G_BL,
                      model = KCBL_synapses.eqs(),
                      on_pre = KCBL_synapses.onpre(),
                      on_post = KCBL_synapses.onpost(),
                      namespace = KCBL_synapses.namespace(),
                      method = KCBL_synapses.method())

    if train:
        S_KCBL.connect(p = conn_params['PKCBL'])
    else:
        S_KCBL_conn = conn_params['S_KCBL_conn']
        S_KCBL.connect(i = S_KCBL_conn['i'], j = S_KCBL_conn['j'])


    S_KCBL.set_states(KCBL_synapses.init_cond())
    S_KCBL.delay = KCBL_synapses.getDelay()

    net.add(S_ALKC, S_KCGGN, S_GGNKC, S_KCBL)

    return [S_ALKC, S_KCGGN, S_GGNKC, S_KCBL]





def get_AL(AL_params, net, train = True):
    
    N_AL = AL_params['N']
    g_syn = AL_params['g_syn']
    
    AL_neuron = AL_params['neuron_class'](AL_params['mon'])
    AL_synapses = AL_params['syn_class'](g_syn)
    
    eqns_neuron = AL_neuron.eqs()


    eqns_syn = AL_synapses.eqs()
    
    G_AL = NeuronGroup(N_AL,
                    model = eqns_neuron,
                    threshold = AL_neuron.threshold(),
                    method = AL_neuron.method(),
                    refractory = AL_neuron.refractory(),
                    reset = AL_neuron.reset(),
                    namespace = AL_neuron.namespace())
    
    trace_AL = StateMonitor(G_AL, AL_neuron.state_mon(), record=True)
    spikes_AL = SpikeMonitor(G_AL)
    
    G_AL.set_states(AL_neuron.init_cond())
    
    S_AL = Synapses(G_AL, G_AL,
                 model = eqns_syn,
                 on_pre = AL_synapses.onpre(),
                 on_post = AL_synapses.onpost(),
                 namespace = AL_synapses.namespace(),
                 method = AL_synapses.method())

    if train:
        S_AL.connect(condition='i != j', p = AL_params['PAL'])
    else:
        S_AL_conn = AL_params['S_AL_conn']
        S_AL.connect(i = S_AL_conn['i'], j = S_AL_conn['j'])

    S_AL.set_states(AL_synapses.init_cond())
    
    net.add(G_AL, S_AL, trace_AL, spikes_AL)
    
    return [G_AL, S_AL, trace_AL, spikes_AL]



def get_KCs(KC_params, net):
    N_KC = KC_params['N']

    KC_neuron = KC_params['neuron_class'](KC_params['mon'])

    G_KC = NeuronGroup( N_KC,
                        model = KC_neuron.eqs(),
                        threshold = KC_neuron.threshold(),
                        reset = KC_neuron.reset(),
                        refractory = KC_neuron.refractory(),
                        method=KC_neuron.method(),
                        namespace = KC_neuron.namespace())


    trace_KC = StateMonitor(G_KC, KC_neuron.state_mon(), record=True)
    spikes_KC = SpikeMonitor(G_KC)

    G_KC.set_states(KC_neuron.init_cond())

    net.add(G_KC, trace_KC, spikes_KC)

    return [G_KC, trace_KC, spikes_KC]



def get_GGN(ggn_params, net):
    GGN_neuron = ggn_params['neuron_class'](ggn_params['mon'])

    G_GGN = NeuronGroup(1,
                        model = GGN_neuron.eqs(),
                        threshold = GGN_neuron.threshold(),
                        reset = GGN_neuron.reset(),
                        refractory = GGN_neuron.refractory(),
                        method = GGN_neuron.method(),
                        namespace = GGN_neuron.namespace())

    trace_GGN = StateMonitor(G_GGN, GGN_neuron.state_mon(), record=True)

    G_GGN.set_states(GGN_neuron.init_cond())
    net.add(G_GGN, trace_GGN)

    return [G_GGN, trace_GGN]




def get_BL(BL_params, net):
    N_BL = BL_params['N']
    g_syn = BL_params['g_syn']

    BL_neuron = BL_params['neuron_class'](BL_params['mon'])
    BL_synapses = BL_params['syn_class'](g_syn)

    G_BL = NeuronGroup( N_BL,
                        model = BL_neuron.eqs(),
                        threshold = BL_neuron.threshold(),
                        reset = BL_neuron.reset(),
                        refractory = BL_neuron.refractory(),
                        method=BL_neuron.method(),
                        namespace = BL_neuron.namespace())

    
    trace_BL = StateMonitor(G_BL, BL_neuron.state_mon(), record=True)
    spikes_BL = SpikeMonitor(G_BL)

    G_BL.set_states(BL_neuron.init_cond())

    S_BL = Synapses(G_BL, G_BL,
             model = BL_synapses.eqs(),
             on_pre = BL_synapses.onpre(),
             on_post = BL_synapses.onpost(),
             namespace = BL_synapses.namespace(),
             method = BL_synapses.method())
    
    S_BL.connect(condition='i != j')

    S_BL.set_states(BL_synapses.init_cond())

    net.add(G_BL, S_BL, trace_BL, spikes_BL)

    return [G_BL, S_BL, trace_BL, spikes_BL]
