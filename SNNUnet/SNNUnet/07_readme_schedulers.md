# Readme file for the Simplified version of the NN-Unet 


The scripts to which this readme referes to are implemented in: ${\color{green}\text{"Scheduler"}}$

As it can be seen, the <span style="color:green;">"Scheduler"</span> contains following scripts:

```
    Losses:
        --> fibonacci_scheduler.py
        --> polylr.py
```

Each of the schedulers implement its own version of the learning rate scheduling. They have also a code names to easy access each of the schedulers in the training scipts: <span style="color:green;"><b>fibo</b> for fiboancci_scheduler</span> and <span style="color:green;"><b>poly</b> for polynomial_scheduler</span>. The polynomial scheduler is the one used in the NNUnet, but it is modify to comply with current guidance of scheduler implementation in torch. On the other hand, fibonacci scheduler is implemented from scratch in a pursiut to give better learning rate scheduling. Learnin rate scheduling is basically setting the learning rate to precaluclated value a te the begining of the each epoch.

---

### polylr.py script:

<details>

<span style="color:green;"> "polylr.py"</span> located in <span style="color:green;"> "Scheduler"</span> directory contains implementation of the <b>polynomial learning rate scheduler</b> same as in the NNUnet. New learning rate is calculated based on the following formula:

$\displaystyle \alpha(t) = \alpha_{0}\,[\max(0,\,1 - t/T)]^{p}$

where $t$ is current step, $T$ is maximum number of steps (number of epochs), $p$ is exponent typically set to 0.9, and $\alpha_0$ is the initial learning rate. The mehtod implemets necessary methods according to Torch guidance:

<ul>
<li>"__init__" -- method which intilise scheduler and inherits <b>_LRScheduler</b> from Torch. Information about scheduler can be found on the following <a href=https://pytorch.org/docs/stable/optim.html> link </a>. The parameters are as follows:</li>
<ul>
<li>optimizer -- torch.optimizer -- optimizer on which the selected scheduler is being applied. </li>
<li>initial_lr -- float -- start learning rate.</li>
<li>max_steps -- int -- maximum number of epochs.</li>
<li>exponent -- float -- number which dictates the slope of the polynomial curve.</li>
<li>current_step -- int -- current step in order we want to resume training. Remember, step is equal to the epoch.</li>
</ul>
<li>"get_lr" -- method which calculates the current learning rate based on the current state of the scheduler(mainly current epoch).</li>
<li>"step" -- updates the scheduler to a new state (new learning rate calculation).</li>
<li>"state_dict" -- method which enables saving of the scheduler's state in case of resuming the training from the checkpoint.</li>
<li>"load_state_dict" -- method which loads the scheduler's state in case of resuming the training from the checkpoint.</li>

</ul>


Basically this implementation is in accordance to torch guides, it inherits the lrscheduler from torch and implements two crucial methods <b>get\_lr</b> and <b>step</b> which are in charge of calculating the next learning rate and updating the scheduler which then updates the optimizer.

</details>


---

### fibbonaci_scheduler.py script

<details>

<span style="color:green;"> "fibonnaci_scheduler.py"</span> located in <span style="color:green;"> "Scheduler"</span> directory contains implementation of the fibbonaci based learning rate scheduler. It is an atempt to schduler learning rate more efectively because the polynomial scheduler keeps learning rate relatively high and the it drops it to 0. Idea was to do arches of different length which will keep learning rates for more time in "higher" values but still spend enough "time" approaching the 0. Here are the parameters of method:

<b>Args</b>:
<ul>
<li>optimizer -- torch.optimizer -- optimizer on which the selected scheduler is being applied.</li>
<li>initial_lr -- float -- start learning rate.</li>
<li>max_epoch -- int -- maximum number of epochs.</li>
<li>arch_p -- float -- minimum required number of epochs in every arch calculated as arch_p*max_epochs.</li>
<li>last_epoch -- int -- last epoch in case that we are resuming the training.</li>
</ul>

To understand how it is implemented, here is a little story:

Method creates creates arches where every arch is <b>negative ln function</b> with given constraints. Arches are long k epochs where k is based on fibonacci sequence (1,1,2,3,5...). k=1 represents n epochs where n is given as a  constraint which contains at least p_arch percent of max epochs. 

For instance: if max epochs is set to be 1000, and p_arch is set to 10% which means that arch
must have at least 0.1*1000=100 epochs. Last sequence which satisfies this constraint is 0,1,1,2,3 because 0+1+1+2+3 = 7, 1000/7 ~ 143. In case that we get one more fibo seqence element 0+1+1+2+3+5 = 12, 1000/12 ~ 83 --> fails constraint of 10% arch lenght. 

Each arch is mimicing function: f(x)= a * (-np.log(x + b) + c where a is fixed to 1.0, b and c
are calculated such that the function satisfies following constraint f(0) = 5, f(arch_end) = 0. This constraint was set based on experiments conducted. The beggining and end learning rates of the arch are then mapped to the function (simple 
proportion mapping)

The beggining and end learning rates of each arch are linearly caluclated. E.G. if begining lr 
is 1e-2 and last one is 0 (always is last set to 0), and we have 5 arches (0-1,1-1, 1-2, 2-3) 
lrs are as follows [0.01   0.0075 0.005  0.0025 0.]. Arch lengts in epochs 
[0, 142, 284, 568, 1000] --> first arch is from 0 to 142, second is 142-284, third is 284-256, 
and last one is 568-1000.

Methods which are used in this scheduler are as follows:

<ul>
<li>"__init__" -- init method which sets all parameters and caluclate learning rates. Learning rates are calculated with help of <b>LearningRateValueEstimation</b> class. This class as a final product returns list of where every index is a learning rate for the epoch of that index (<b>method obtain_learning_rates</b>).</li>
<li>"get_lr" -- method which calculates current lrearning rate based on the current step(epoch).</li>
<li>"state_dict" -- method which enables saving of the scheduler's state in case of resuming the training from the checkpoint.</li>
<li>"load_state_dict" -- method which loads the scheduler's state in case of resuming the training from the checkpoint.</li>
</ul>

How does it work? Well, it is mostly the same as any torch scheduler, but it calculates the learning rates based on the provided <b>LearningRateValueEstimation</b> class. This class is the "brain" of the whole scheduler. It has four methods:

<ul>
<li>"__fibo_sequence" -- method which calculates number of elements in fibonnaci sequence such that constraint of each sequence (arch) is haveing at least <b>arch_p</b> precented of epoch in it. Lower arch_p means less epochs in every arch and more arches. E.G. for number of fibonnaci sequence set to 5 (1,1,2,3,5) it contains total 1+1+2+3+5 = 12 divisions, and if we have 500 epochs it means that each arch will be 500/12 = 41 long (which satisfies constraint of 0.08% for arch_p. If we set arch_p to like 10%, the sequence will be 1,1,2,3 because that is arch length of 71 epochs which is 0.14%)</li>
<li>"__obtain_interim_learning_rate" -- each arch/sequence has its beggining and ending learning rate. This learning rates are calculated based on the linear interpolation of number of arches over the number of epochs. E.G. if there are 4 arches and 1000 epochs in total, the initial learning rate is given as <b>initial_lr</b> while the ending learning rate is set to 0. Then 1000 points will be sampled from interval <b>intial_lr</b> and 0 where in obtain_learning_rates those learning rates which are on the corresponding epochs of the arches begining and ends will be selected as strating and ending points of function given in <b>remodel_function</b></li>
<li>"rolemodel_function" -- function which connects beggining and end point of an arch. This function actually is an arch. We have used <b>f(x) = a * (-np.log(x + b) + c</b> with parameter a fixed to 0.1. Paramers b and c are calculated based on the given two points of an arch.</li>
<li>"obtain_learning_rates" -- method which finally caculates larning rate for each epoch by calling <b>rolemodel_function</b> with adequate parameters. Ultiamately, this provides a list of learning rates.</li>
</ul>

</details>