import operator
import time

import litenn as nn
import litenn.core as nc


def backward(loss, stop_grad=None, grad_for_non_trainables=False, keep_gradient_for_non_trainables=False):
    """
    Perform backward computation.

    arguments:

        loss

    Tensor or list of Tensors from which backprop should start.
    If value is dict, then keys are loss, and values are numpy gradient init.

        stop_grad (None)

    None or Tensor or list of Tensors where backprop should stop.

        grad_for_non_trainables (False)

    If False, the backprop will stop at those branches that have no trainable tensors (not attached to any Optimizer),
    also gradient for intermediate non trainable tensors will be freed in order to reduce memory consumption.
    """

    loss_gradients = {}
    if isinstance(loss, dict):
        loss_gradients = loss
        loss = list(loss.keys())
    elif not isinstance(loss, (tuple,list) ):
        loss = [loss]

    if stop_grad is None:
        stop_grad = []
    elif not isinstance(stop_grad, (tuple,list) ):
        stop_grad = [stop_grad]
    user_stop_grad = set(stop_grad)

    key_get_seq_id = operator.methodcaller('_get_seq_id')

    # key is Tensor._get_seq_id(),
    # value is list[] of  Tensor._get_seq_id() which are produced by key-tensor
    stop_grad = {}
    
    if not grad_for_non_trainables:
        # Determine which branches don't require grad
        # Fill stop_grad dict.

        # form working_list from current loss list
        working_list = loss.copy()
        for t in working_list:
            # loss tensors produce nothing, so set empty array
            stop_grad[t._get_seq_id()] = []

        # Go backward in working_list
        while len(working_list) != 0:
            # Process tensor with largest _get_seq_id (most late)
            working_list = sorted(working_list, key=key_get_seq_id)
            t = working_list.pop(-1)
            t_seq_id =  t._get_seq_id()

            if not t._is_trainable():
                # Tensor is not attached to Optimizer (not marked as _is_trainable)
                t_gradfns = t._get_gradfns()
                if t_gradfns is not None:
                    # Tensor is produced by input tensors, iterate over them
                    for input_t in t_gradfns:
                        input_seq_id = input_t._get_seq_id()

                        # Get/set array of next tensors for input_t key
                        i_nexts = stop_grad.get(input_seq_id, None)
                        if i_nexts is None:
                            stop_grad[input_seq_id] = i_nexts = []

                        if t_seq_id not in i_nexts:
                            # add Tensor to array of next tensors
                            i_nexts.append(t_seq_id)

                        if input_t._is_reference():
                            input_t = input_t._get_top_reference_source()

                        if input_t not in working_list:
                            # add input_t to current working_list
                            working_list.append(input_t)
            else:
                # Tensor is attached to Optimizer (marked as _is_trainable
                # Go forward and remove all tensors from stop_grad
                # which produce this and next tensors
                t_list = [t._get_seq_id()]
                while len(t_list) != 0:
                    t = t_list.pop(-1)
                    t_nexts = stop_grad.get(t, None)
                    if t_nexts is not None:
                        t_list += t_nexts
                    if t in stop_grad:
                        stop_grad.pop(t)

    # add user stop_grad
    for t in user_stop_grad:
        stop_grad[t._get_seq_id()] = []

    # Remove tensors from loss list if they are in stop_grad
    loss_list = [t for t in loss if t._get_seq_id() not in stop_grad]

    # Set initial gradient for tensors in loss_list
    for t in loss_list:
        t_grad = t.get_grad()
        
        if t in loss_gradients:
            # set value specified by user
            nc.op.add(t_grad, nn.Tensor_from_value(loss_gradients[t]), output_t=t_grad )
        else:
            nc.op.add_const(t_grad, 1.0, output_t=t_grad)

    # Convert reference tensors in loss_list to their source
    loss_list = [t._get_top_reference_source() if t._is_reference() else t for t in loss_list]
    # Filter duplicates
    loss_list = list(set(loss_list)) 

    timings = []

    # Go backward in working loss_list
    while len(loss_list) != 0:
        # Process tensor with largest _get_seq_id (most late)
        loss_list = sorted(loss_list, key=key_get_seq_id) 
        t = loss_list.pop(-1)

        t_gradfns = t._get_gradfns()
        if t_gradfns is not None:
            # Tensor is produced by input tensors, iterate over them
            for input_t in t_gradfns:
                if t._is_freezed() and input_t._is_trainable():
                    # if t is under freeze and it's input_t is used by Optimizer (marked as _is_trainable
                    continue # then stop gradient
                if input_t._get_seq_id() in stop_grad:
                    continue

                #nn.devices.wait()
                #tim = time.time()

                # Call gradient computation
                for func in t_gradfns[input_t]:
                    func(t,t.get_grad())

                #nn.devices.wait()
                if input_t._is_reference():
                    input_t = input_t._get_top_reference_source()
                if input_t not in loss_list:
                    # add input_t to working loss_list, if not exist.
                    loss_list.append(input_t)

        if not grad_for_non_trainables and not t._is_trainable():
            # grad_for_non_trainables == False, then
            # we no longer need its gradient if it is not used by Optimizer (marked as _is_trainable
            t.free_grad()

    #timings = sorted(timings, key=operator.itemgetter(0) )

    #print(timings)
    #print(f'sum : { sum( [ x[0] for x in timings] ) }')
    #nn.devices.wait()
    #print(f'time of backward {time.time()-tim}')
