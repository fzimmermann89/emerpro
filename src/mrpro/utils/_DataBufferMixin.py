import itertools
from collections import OrderedDict
from collections.abc import Iterator

import torch

from mrpro.data._Data import Data


class DataBufferMixin(torch.nn.Module):
    """A Mixin that allow to set a mrpro.data.Data as a Buffer.

    This implements the same logic for mrpro.Data as torch.nn.Module
    uses for Buffers, but stores them in a separate dictionary called
    _data

    The main use is to allow Data objects to be automatically moved to a
    devices etc if one calles, for example DataBufferMixin.cuda() will
    call .cuda() also on all register_buffer'ed Data attributes.

    Used in Operators, for example.
    """

    _data: dict[str, Data]
    # could maybe be requiered at some point,
    # see torch.nn.Module
    # call_super_init = True

    def register_buffer(self, name: str, data: torch.Tensor | None | Data, persistent: bool = True) -> None:
        """Add a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter.
        Buffers, by default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`. Data objects always considered non-persistant!

        Buffers can be accessed as attributes using given names.

        See also torch.nn.Module.register_buffer.

        Parameters
        ----------
            name
                name of the buffer. The buffer can be accessed from this module using the given name
            data
                buffer to be registered. If ``None``, then operations that run on buffers, such as `cuda`, are ignored.
                If ``None``, the buffer is **not** included in the module's `state_dict`.
            persistent (bool)
                whether the buffer is part of this module's `state_dict`.
                Data objects are currently never added to the `state_dict`!
        """
        if not isinstance(data, Data):
            return super().register_buffer(name, data, persistent)

        if '_data' not in self.__dict__:
            raise AttributeError('cannot assign buffer before super().__init__() call')
        elif not isinstance(name, str):
            raise TypeError(f'buffer name should be a string. Got {torch.typename(name)}')
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._data:
            raise KeyError(f"attribute '{name}' already exists")
        else:
            # TODO: hooks ?
            self._data[name] = data
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        # Also _apply it to the new _data
        for key, data in self._data.items():
            if data is not None:
                try:
                    # For methods implemented on the Data object
                    self._data[key] = fn(data)
                except AttributeError:
                    # otherwise skip
                    pass

    def __getattr__(self, name: str):
        """Get Attribute."""
        if '_data' in self.__dict__:
            data = self.__dict__['_data']
            if name in data:
                return data[name]
        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        """Set Attribute."""
        if isinstance(value, Data) and (data := self.__dict__.get('_data')) is not None and name in data:
            # TODO: hooks ?
            data[name] = value
        else:
            return super().__setattr__(name, value)

    def __delattr__(self, name):
        """Delete Attribute."""
        if name in self._data:
            del self._data[name]
            self._non_persistent_buffers_set.discard(name)
        else:
            super().__delattr__(name)

    def __init__(self, *args, **kwargs) -> None:
        self._data = OrderedDict()
        super().__init__(*args, **kwargs)

    def named_buffers(
        self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Return an iterator yielding the name of and the buffer itself.

        This also returns mrpro.data.Data derived buffers.

        See also torch.nn.Module.named_buffers

        Parameters
        ----------
            prefix
                prefix to prepend to all buffer names.
            recurse
                if True, then yields buffers of this module and all submodules. Otherwise, yields only buffers that
                are direct members of this module. Defaults to True.
            remove_duplicate
                whether to remove the duplicated buffers in the result. Defaults to True.

        Yields
        ------
            Tuple containing the name and buffer
        """

        buffers = super().named_buffers(prefix, recurse, remove_duplicate)
        data_buffers = self._named_members(
            lambda module: module._data.items(), prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )
        gen = itertools.chain(buffers, data_buffers)
        yield from gen

    def __dir__(self):
        """List Attributes."""
        data = list(self._data.keys())
        keys = super().__dir__() + data
        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]
        return sorted(keys)

    # TODO: implement state_dict logic.
