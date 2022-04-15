import collections
class Logger:
    _editing_mode = False
    
    def __init__(self,name,fathers=[]):
        assert isinstance(name,str)
        fathers = fathers+[name]
        super().__setattr__('_fathers', fathers)
        super().__setattr__('_contents', collections.OrderedDict())
        
    def __repr__(self):
        return 'Logger object: '+'.'.join(self._fathers)
        
    def __str__(self):
        new_line_str = '  \n'
        fathers_str = '.'.join(self._fathers)
        contents_str = new_line_str.join([f'{fathers_str}.{content} = {value}' if not isinstance(value,Logger) else str(value) for content,value in self._contents.items()])
        return contents_str
    
    def __setattr__(self,name,value):
        if __class__._editing_mode:
            assert not name.startswith('_')
            if isinstance(value,str):
                self._contents[name] = f"'{value}'"
            else:
                try:
                    self._contents[name] = value.__name__
                except AttributeError:
                    self._contents[name] = value
                
            super().__setattr__(name, value)
        else:
            raise Exception('can not assign new value outside of editing mode')
    
    def __getattr__(self,name):
        if __class__._editing_mode:
            assert name != 'shape'
            content = Logger(name,self._fathers)
            self._contents[name] = content
            super().__setattr__(name,content)
        else:
            raise AttributeError(f'{repr(self)}.{name} is not defined while in editing mode')
        return getattr(self,name)
    
class Editor:
    def __init__(self,prefix):
        self._prefix = prefix
    def __enter__(self):
        Logger._editing_mode = True
        global Config
        Config = Logger(self._prefix)
        return Config
    def __exit__(self, exc_type, exc_val, exc_tb):
        Logger._editing_mode = False