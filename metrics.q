
accuracy:{[y;r;t]
    sum[sum "j"$("j"$y)="j"$r>=t]%count y
    }

f1score:{[p;r] 2*p*r%(p+r)}

precision:{[y;r;t] sum[sum "j"$y[where raze r>=t]] % sum sum "j"$r>=t}

recall:{[y;r;t] sum[sum "j"$y[where raze r>=t]] % sum sum "j"$y}

logloss:{[y;r] neg[sum sum (y*log[r]) + (1-y)*log 1-r] % count y}

