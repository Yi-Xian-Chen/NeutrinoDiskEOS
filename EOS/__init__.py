""" 
A code to calculate Equation of state quantities (equilibrium Ye, reaction rates and cooling rates etc, as attributes of class NeutrinoDiskEOS) for
Neutrino-cooled collapsar disks, following methods of Beloborodov 2003. The main code takes temperature and 
density arrays as inputs and generate thermodynamic quantities as functions of temperature and density. Extra modules
of the code can invert the output to generate cooling rate as functions of pressure and density etc, for use of 
hydrodynamic simulations.
"""

__version__ = "0.0.1"
__author__ = "Yixian Chen"
__email__ = "yc9993@princeton.edu"
__all__ = ["help_info", "NeutrinoDiskEOS"]


def help_info():
    """ Print Basic Help Info """

    print("""
    **************************************************************************
    * 
    * A code to calculate Equation of state quantities (equilibrium Ye, reaction rates and cooling rates etc) for
    * Neutrino-cooled collapsar disks, following methods of Beloborodov 2003. The main code takes temperature and 
    * density arrays as inputs and generate thermodynamic quantities as functions of temperature and density. Extra modules
    * of the code can invert the output to generate cooling rate as functions of pressure and density etc, for use of 
    * hydrodynamic simulations.
    * 
    * Author: Yixian Chen
    * Current Version: 0.0.1
    * Note: This package is still under active development and we welcome
    *       any comments and suggestions.
    **************************************************************************
    """)