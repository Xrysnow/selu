import mxnet as mx
import numpy as np


class SELU(mx.operator.CustomOp):
    def __init__(self):
        self.alpha=1.6732632423543772848170429916717
        self.scale=1.0507009873554804934193349852946

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        xp = (x + mx.nd.abs(x)) / 2
        xn = x - xp
        y = self.scale * (self.alpha * (mx.nd.exp(xn) - 1) + xp)
        self.assign(out_data[0], req[0], y)
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = in_data[0]
        tg = out_grad[0]
        pos = (x >= 0)
        xn = (x - mx.nd.abs(x)) / 2
        in_grad[0][:] = (self.scale * ((1 - self.alpha) * pos + self.alpha * mx.nd.exp(xn))) * tg

@mx.operator.register("selu")
class SELUProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SELUProp, self).__init__(need_top_grad=True)
    
    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        shape = in_shape[0]
        return [shape], [shape]

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return SELU()
    
if __name__ == '__main__':
    
    data = mx.sym.var('data')
    net = mx.symbol.FullyConnected(data=data,name="fc1",num_hidden=2)
    net = mx.sym.Custom(net, name = 'snn1', op_type = 'selu')
    net = mx.sym.LogisticRegressionOutput(data=net, name = 'lr')
