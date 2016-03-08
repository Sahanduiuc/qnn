sigmoid:{1f % 1f + exp neg x}
perceptron:{sigmoid x mmu y}
addbias:{{1f,x} each x}
exceptbias:{x[;1+til count[first x]-1]}
layer:{perceptron[addbias x;flip y]}
predict:{[x;m] last x layer\ m}

/initlayer: randomly initialize layer weights
/x - number of inputs in current layer
/y - number of outputs in current layer (except "bias") 
initlayer:{[x;y]
    eps:sqrt[6f] % sqrt x + y + 1;
    rv:{rand each x} each (y;(1+x))#1f;
    neg[eps]+2*eps*rv
    }

/init model: randomly initialize weights in all layers
/x - array with number of units in each layer (excpet "bias" units)
/first x - number of inputs of the network (i.e. number of features)
/last x - number of outputs of the network (i.e. number of classes)
/e.g.: 
/x:400 25 10 - network with 400 inputs (features), 25 units in one hidden layer and 10 outputs
initmodel:{{initlayer[first x;last x]} each {$[2=count x; enlist x; enlist[2#x],.z.s 1_x]} x}

/logcost: compute log cost function
/r - real network outputs
/y - expected network outputs
/m - model parametes (weights)
/l - regularization parameter
logcost:{[r;y;m;l]
    sqsum:{s:sum sum exceptbias x; s*s};
    cost:sum sum neg[y*log r]-(1-y)*log[1-r];
    cost +: l * 0.5f * sum sqsum each m;
    cost % count y
    }

/backprop
/rlo - revrse[lo],enlist x
/y - expected outputs
/rm - revese m
/l - regularization parameter
backprop:{[rlo;y;rm;l]
    ld:first[rlo] - y; /last delta
    /calc deltas
    dts:enlist[ld],{[nd;m;lo] exceptbias[nd mmu m] * lo * 1-lo} \ [ld;rm[til count[rm]-1];rlo[1+til count [rlo]-2]];
    /gradients
    grads:{[dts;lo] flip[dts] mmu addbias lo}'[dts;1_rlo];
    /regularization 
    reg:{[m;l] {0f,x} each l*exceptbias m};
    reverse (grads + reg'[rm;l])%count y
    }

/gradient step
gradstep:{[x;y;m;l]
    lo:x layer\ m;
    cost:logcost[last lo;y;m;l];
    cost,backprop[reverse[lo],enlist x;y;reverse m;l]
    }

/gradient descent with adaptive learnin rate/step size
asgd:{[x;y;m;l]
    lr:0.1;      /initial learning rate
    irp:1.1;     /learning rate increase parameter
    drp:0.5;     /learning rate decrease parameter
    thrsh:1e-16; /threshold
    gr:gradstep[x;y;m;l];
    pcost:first gr;
    cm: m - lr * 1_gr;
    while [(lr > thrsh%2) and thrsh < abs pcost - first gr:gradstep[x;y;cm;l];
        cost:first gr;
        if [cost > pcost; lr *: drp];
        if [cost < pcost; 
            lr *: irp;
            cm -: lr * 1_gr; 
            pcost:cost
            ];
        ];
    cm
    }

/stochastic gradient descent
sgd:{[x;y;m;l]
    lr:0.1;
    irp:1.1;
    drp:0.5;
    thrsh:1e-16;
    ecnt:5;
    gr:gradstep[x;y;m;l];
    pcost:first gr;
    cm: m - lr * 1_gr;
    do[ecnt;
        xi:0;
        do[count y;
            gr:gradstep[enlist x xi;enlist y xi;cm;l];
            xi+:1;
            cost:first gr;
            if [(lr < thrsh%2) and thrsh > abs pcost - cost; :cm];
            if [cost > pcost; lr *: drp];
            if [cost < pcost;
                lr *: irp;
                cm -: lr * 1_gr;
                pcost:cost;
                ];
            ];
        ];
    cm
    }


/mulrand: multiple random inititalization and select best one
mulrand:{[x;y;mp;l;c]
    models:([] modl:();res:());
    do[c;
        model:asgd[X;Y;initmodel mp;l];
        res:predict[X;model] - Y;
        res:sum sum(res*res);
        models,:enlist (model;res);
        ];
        :first exec modl from models where res=min res;
    }

//
X:"f"$(0 0; 0 1; 1 0; 1 1)
0N!`X,X;

//OR
mp: 2 1
0N!`OR,enlist mp;
Y:enlist each (0 1 1 1f)
predict[X;asgd[X;Y;initmodel mp;0]]
//AND
mp:2 1
0N!`AND,enlist mp;
Y:enlist each (0 0 0 1f)
predict[X;asgd[X;Y;initmodel mp;0]]
//XOR
mp:2 2 1
0N!`XOR,enlist[mp],enlist "Multiple init";
Y:enlist each (0 1 1 0f)
predict[X;mulrand[X;Y;mp;0;100]]


