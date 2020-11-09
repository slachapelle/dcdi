"""
Modified from:
.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""
import os
import uuid
import warnings
import networkx as nx
from shutil import rmtree
from cdt.causality.graph.model import GraphModel
from pandas import DataFrame, read_csv
from cdt.utils.Settings import SETTINGS
from cdt.utils.R import RPackages, launch_R_script
import numpy as np
import torch


def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'

warnings.formatwarning = message_warning


class PC(GraphModel):
    """JCI-PC algorithm **[R model]**.

    Args:
        verbose (bool): Defaults to ``cdt.SETTINGS.verbose``.
    """

    def __init__(self, verbose=False):
        """Init the model and its available arguments."""
        if not RPackages.pcalg:
            raise ImportError("R Package pcalg is not available.")

        super().__init__()
        self.arguments = {'{FOLDER}': '/tmp/cdt_pc/',
                          '{FILE}': 'data.csv',
                          '{SKELETON}': 'FALSE',
                          '{GAPS}': 'fixedgaps.csv',
                          '{REGIMES}': 'regimes.csv',
                          '{TARGETS}': 'targets.csv',
                          '{VERBOSE}': 'FALSE',
                          '{ALPHA}': '1e-2',
                          '{OUTPUT}': 'result.csv'}
        self.verbose = SETTINGS.get_default(verbose=verbose)


    def _run_pc(self, data, fixedGaps=None, regimes=None, alpha=None,
                indep_test="gaussCItest", known=False, targets=None, verbose=True):
        """Setting up and running JCI-PC with all arguments."""
        id = str(uuid.uuid4())
        os.makedirs('/tmp/cdt_pc' + id + '/')
        self.arguments['{FOLDER}'] = '/tmp/cdt_pc' + id + '/'
        self.arguments['{ALPHA}'] = str(alpha)
        self.arguments['{INDEP_TEST}'] = indep_test
        if known:
            self.arguments['{KNOWN}'] = 'TRUE'
        else:
            self.arguments['{KNOWN}'] = 'FALSE'

        def retrieve_result():
            return read_csv('/tmp/cdt_pc' + id + '/result.csv', delimiter=',').values

        try:
            data.to_csv('/tmp/cdt_pc' + id + '/data.csv', header=False, index=False)
            if targets is not None:
                np.savetxt('/tmp/cdt_pc' + id + '/targets.csv', targets, delimiter=",")

            if fixedGaps is not None:
                fixedGaps.to_csv('/tmp/cdt_pc' + id + '/fixedgaps.csv', index=False, header=False)
                self.arguments['{SKELETON}'] = 'TRUE'
            else:
                self.arguments['{SKELETON}'] = 'FALSE'

            if regimes is not None:
                regimes.to_csv('/tmp/cdt_pc' + id + '/regimes.csv', index=False, header=False)
                self.arguments['{INTERVENTION}'] = 'TRUE'
            else:
                self.arguments['{INTERVENTION}'] = 'FALSE'

            pc_results = launch_R_script("{}/pc.R".format(os.path.dirname(os.path.realpath(__file__))), self.arguments, output_function=retrieve_result, verbose=verbose)
            print(pc_results)

        # Cleanup
        except Exception as e:
            rmtree('/tmp/cdt_pc' + id + '')
            raise e
        except KeyboardInterrupt:
            rmtree('/tmp/cdt_pc' + id + '/')
            raise KeyboardInterrupt
        rmtree('/tmp/cdt_pc' + id + '')
        return pc_results
