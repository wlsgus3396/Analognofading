# Load sub-packages
import models.vampyre.common
import models.vampyre.estim
import models.vampyre.trans
import models.vampyre.solver

def version():
    """
        Return the current version string for the vampyre package.
        
        >>> version()
        '0.0'
    """
    return "0.0"

def version_info():
    print("vampyre version " + version())