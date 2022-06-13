from . import metrics

def MetricFuncionParser(metric_funcs):
    metric_funcs = [eval(f'metrics.{m}') for m in metric_funcs]
    return MetricFunction(metric_funcs)

class MetricFunction:
    def __init__(self,metric_funcs):
        self.metric_funcs = metric_funcs
        
    def __iter__(self):
        return iter(self.metric_funcs)
        
    def __call__(self,output,data):
        metric = {}
        for m in self.metric_funcs:
            acc_count, acc_metric = m(output,data)
            metric[m.name] = {'acc_count':acc_count.item(),'acc_value':acc_metric.item()}
        return metric

class MetricBuffer:
    def __init__(self,names):
        self.buffers = {n:{'acc_count':0,'acc_metric':0} for n in names}
    
    def add(self,metric_values):
        for name, metric in metric_values.items():
            self.buffers[name]['acc_count'] += metric['acc_count']
            self.buffers[name]['acc_metric'] += metric['acc_metric']
            
    def result(self):
        return {n: m['acc_metric']/m['acc_count'] if m['acc_count'] != 0 else None for n,m in self.buffers.items()}
    
    def clear(self):
        for n in self.buffers.keys():
            self.buffers[n]['acc_count'] = 0
            self.buffers[n]['acc_metric'] = 0