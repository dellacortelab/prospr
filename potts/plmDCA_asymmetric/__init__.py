#Copyright 2015-2017 MathWorks, Inc.

# This template is used to generate an __init__.py file for a particular deployable package.

""" Package for executing deployed MATLAB functions """

from __future__ import print_function
import atexit
import glob
import importlib
import os
import os.path
import pdb
import platform
import re
import sys
import weakref

class _PathInitializer(object):
    PLATFORM_DICT = {'Windows': ['PATH','dll',''], 'Linux': ['LD_LIBRARY_PATH','so','libmw'], 'Darwin': ['DYLD_LIBRARY_PATH','dylib','libmw']}
    SUPPORTED_PYTHON_VERSIONS = ['2_7', '3_5', '3_6']
    RUNTIME_VERSION_W_DOTS = '9.4'
    RUNTIME_VERSION_W_UNDERSCORES = '9_4'
    PACKAGE_NAME = 'plmDCA_asymmetric'
    
    def set_interpreter_version(self):    
        """Make sure the interpreter version is supported."""
        ver = sys.version_info
        version = '{0}_{1}'.format(ver[0], ver[1])

        if version in _PathInitializer.SUPPORTED_PYTHON_VERSIONS:
            self.interpreter_version = version
        else:
            version_with_dot = version.replace("_", ".")
            raise EnvironmentError("Python {0} is not supported.".format(version_with_dot))

    def __init__(self):
        """Initialize the variables."""
        self.arch = ''
        self.is_linux = False    
        self.is_mac = False
        self.is_windows = False
        self.mr_handle = None
        self.ml_handle = None
        self.system = ''
        self.cppext_handle = None

        # path to the folder that stores Python extensions and shared libraries
        self.bin_dir = ''

        # path to the folder that stores pure Python matlab_pysdk.runtime code (_runtime_dir)
        self.runtime_dir = ''

        # path to the folder that stores the pure Python matlab mlarray code used for type conversion
        self.ml_dir = ''

        self.set_interpreter_version()
        self.get_platform_info()

        this_folder = os.path.dirname(os.path.realpath(__file__))
        self.path_file_name = os.path.join(this_folder, 'paths.{0}.txt'.format(self.arch))

        self.instances_of_this_package = set([])

            
    def read_path_file(self):
        """Look for a file that lists items to add to path. If present, read it and add the paths."""
        filtered_lines = []
        if os.path.isfile(self.path_file_name):
            pth_file = open(self.path_file_name, 'r')
            lines = pth_file.readlines()
            for line in lines:
                stripped_line = line.strip()
                if stripped_line and stripped_line[0] != '#':
                    filtered_lines.append(stripped_line)
        length = len(filtered_lines)
        if length == 3:
            (bin_dir, runtime_dir, ml_dir) = filtered_lines
            if (not os.path.isdir(bin_dir)) or (not os.path.isdir(runtime_dir)) or (not os.path.isdir(ml_dir)):
                return False
            else:
                (self.bin_dir, self.runtime_dir, self.ml_dir) = (bin_dir, runtime_dir, ml_dir)
                return True
        else:
            return False

    def write_path_file(self):
        """Write a file that lists items to add to path. If present, it will be overwritten."""
        existing_contents = ''
        if os.path.isfile(self.path_file_name):
            path_file = open(self.path_file_name, 'r')
            existing_contents = path_file.readlines()
            path_file.close()

        path_file = open(self.path_file_name, 'w')
        if self.system == 'Windows':
            print('# bin dir: added to both OS path and system path', file=path_file)
        else:
            print('# bin dir: added to system path', file=path_file)
        print(self.bin_dir, file=path_file)
        print('', file=path_file)
        
        print('# runtime dir: added to system path', file=path_file)
        print(self.runtime_dir, file=path_file)
        print('', file=path_file)

        print('# matlab (mlarray) dir: added to system path', file=path_file)
        print(self.ml_dir, file=path_file)
        print('', file=path_file)

        if existing_contents:
            print(existing_contents, file=path_file)
        path_file.close()
        
    def get_platform_info(self):
        """Ask Python for the platform and architecture."""
    
        # This will return 'Windows', 'Linux', or 'Darwin' (for Mac).
        self.system = platform.system() 
        if not self.system in _PathInitializer.PLATFORM_DICT:
            raise RuntimeError('{0} is not a supported platform.'.format(self.system))
        else:
            # path_var is the OS-dependent name of the path variable ('PATH', 'LD_LIBRARY_PATH', "DYLD_LIBRARY_PATH')
            (self.path_var, self.ext, self.lib_prefix) = _PathInitializer.PLATFORM_DICT[self.system]

        if self.system == 'Windows':
            self.is_windows = True
            bit_str = platform.architecture()[0]
            if bit_str == '64bit':
                self.arch = 'win64'
            elif bit_str == '32bit':
                self.arch = 'win32'
            else:
                raise RuntimeError('{0} is not supported.'.format(bit_str))
        elif self.system == 'Linux':
            self.is_linux = True
            self.arch = 'glnxa64'
        elif self.system == 'Darwin':
            self.is_mac = True
            self.arch = 'maci64'
        else:
            raise RuntimeError('Operating system {0} is not supported.'.format(self.system))
        
    def get_paths_from_os(self):
        """ 
        Look through the system path for a file whose name contains a runtime version
        corresponding to the one with which this package was produced.
        """

        # Concatenates the pieces into a string. The double parentheses are necessary.
        if self.system == 'Windows':
            file_to_find = ''.join((self.lib_prefix, 'mclmcrrt',
                 _PathInitializer.RUNTIME_VERSION_W_UNDERSCORES, '.', self.ext))
        elif self.system == 'Linux':
            file_to_find = ''.join((self.lib_prefix, 'mclmcrrt', '.', self.ext, '.',
                                    _PathInitializer.RUNTIME_VERSION_W_DOTS))
        elif self.system == 'Darwin':
            file_to_find = ''.join((self.lib_prefix, 'mclmcrrt', '.', 
                                    _PathInitializer.RUNTIME_VERSION_W_DOTS,
                                    '.', self.ext))
        else:
            raise RuntimeError('Operating system {0} is not supported.'.format(self.system))

        path_elements = []
        if self.path_var in os.environ:
            path_elements = os.environ[self.path_var].split(os.pathsep)
        if not path_elements:
            friendly_os_name = self.system
            if friendly_os_name == 'Darwin':
                friendly_os_name = 'Mac'
            raise RuntimeError('On {0}, you must set the environment variable "{1}" to a non-empty string. {2}'.format(
                friendly_os_name, self.path_var, 'For more details, see the package documentation.'))

        path_found = ''
        for elem in path_elements:
            filename = os.path.join(elem, file_to_find)
            if (os.path.isfile(filename)):
                path_found = elem
                break
        if not path_found:
            raise RuntimeError('Could not find an appropriate directory for MATLAB or the MATLAB runtime in {0}. Details: {1}'.format(
                self.path_var, file_to_find))

        path_components = re.split(r'\\|/', path_found)
        
        if path_components[-1]:
            last_path_component = path_components[-1]
        else:
            # The directory name ended with a slash, so the last item in the list was an empty string. Go back one more.
            last_path_component = path_components[-2]

        if last_path_component != self.arch:
            output_str = ''.join(('To call deployed MATLAB code on a {0} machine, you must run a {0} version of Python, ',
                'and your {1} variable must contain an element pointing to "<MR>{2}runtime{2}{0}", ',
                'where "<MR>" indicates a MATLAB or MATLAB Runtime root. ',
                'Instead, the value found was as follows: {3}'))
            raise RuntimeError(output_str.format(self.arch, self.path_var, os.sep, path_found))
            
        matlabroot = os.path.dirname(os.path.dirname(os.path.normpath(path_found)))
        bin_dir = os.path.join(matlabroot, 'bin', self.arch)
        runtime_dir = os.path.join(matlabroot, 'toolbox', 'compiler_sdk', 'pysdk_py')
        ml_dir = os.path.join(runtime_dir, 'mlarray_dist')
        if not os.path.isdir(bin_dir):
            raise RuntimeError('Could not find the directory {0}'.format(bin_dir))
        if not os.path.isdir(runtime_dir):
            raise RuntimeError('Could not find the directory {0}'.format(runtime_dir))
        if not os.path.isdir(ml_dir):
            raise RuntimeError('Could not find the directory {0}'.format(ml_dir))
        (self.bin_dir, self.runtime_dir, self.ml_dir) = (bin_dir, runtime_dir, ml_dir)

    def update_paths(self):
        """Update the OS and Python paths."""

        #For Windows, add the bin_dir to the OS path. This is unnecessary
        #for Linux and Mac, where the OS can find this information via rpath.
        if self.is_windows:
            os.environ[self.path_var] = self.bin_dir + os.pathsep + os.environ[self.path_var]

        #Add all paths to the Python path.
        sys.path.insert(0, self.bin_dir)
        sys.path.insert(0, self.runtime_dir)
        sys.path.insert(0, self.ml_dir)

    def import_matlab_pysdk_runtime(self):
        """Import matlab_pysdk.runtime. Must be done after update_paths() and import_cppext() are called."""
        try:
            self.mr_handle = importlib.import_module('matlab_pysdk.runtime')
        except Exception as e:
            raise e

        if not hasattr(self.mr_handle, '_runtime_version_w_dots'):
            raise RuntimeError('Runtime version of package ({0}) does not match runtime version of previously loaded package'.format(
                _PathInitializer.RUNTIME_VERSION_W_DOTS))
        elif self.mr_handle._runtime_version_w_dots and (self.mr_handle._runtime_version_w_dots != _PathInitializer.RUNTIME_VERSION_W_DOTS):
            raise RuntimeError('Runtime version of package ({0}) does not match runtime version of previously loaded package ({1})'.format(
                _PathInitializer.RUNTIME_VERSION_W_DOTS,
                self.mr_handle._runtime_version_w_dots))
        else:
            self.mr_handle._runtime_version_w_dots = _PathInitializer.RUNTIME_VERSION_W_DOTS

        self.mr_handle._cppext_handle = self.cppext_handle

    def import_matlab(self):
        """Import the matlab package. Must be done after Python system path contains what it needs to."""
        try:
            self.ml_handle = importlib.import_module('matlab')
        except Exception as e:
            raise e

    def initialize_package(self):
        package_handle = self.mr_handle.DeployablePackage(self, self.PACKAGE_NAME, __file__)
        self.instances_of_this_package.add(weakref.ref(package_handle))
        package_handle.initialize()
        return package_handle

    def initialize_runtime(self, option_list):
        if not self.cppext_handle:
            raise RuntimeError('Cannot call initialize_application before import_cppext.')
        if self.is_mac:
            ignored_option_found = False
            for option in option_list:
                if option in ('-nodisplay', '-nojvm'):
                    ignored_option_found = True
                    break
            if ignored_option_found:
                print('WARNING: Options "-nodisplay" and "-nojvm" are ignored on Mac.')
                print('They must be passed to mwpython in order to take effect.')
        self.cppext_handle.initializeApplication(option_list)

    def terminate_runtime(self):
        if not self.cppext_handle:
            raise RuntimeError('Cannot call terminate_application before import_cppext.')
        self.cppext_handle.terminateApplication()

    def import_cppext(self):
        self.cppext_handle = importlib.import_module("matlabruntimeforpython" + self.interpreter_version)

try:
    _pir = _PathInitializer()
    _pir.get_paths_from_os()

    _pir.update_paths()
    _pir.import_cppext()
    _pir.import_matlab_pysdk_runtime()
    _pir.import_matlab()
except Exception as e:
    print("Exception caught during initialization of Python interface. Details: {0}".format(e))
    raise
    # We let the program exit normally.

def initialize():
    """ 
    Initialize package and return a handle.

    Initialize a package consisting of one or more deployed MATLAB functions. The return
    value is used as a handle on which any of the functions can be executed. To wait
    for all graphical figures to close before continuing, call wait_for_figures_to_close() 
    on the handle. To close the package, call terminate(), quit() or exit() (which are 
    synonymous) on the handle. The terminate() function is executed automatically when the 
    script or session ends.

    Returns
        handle - used to execute deployed MATLAB functions and to call terminate()
    """
    return _pir.initialize_package()

def initialize_runtime(option_list):
    """
    Initialize runtime with a list of startup options.

    Initialize the MATLAB Runtime with a list of startup options that will affect 
    all packages opened within the script or session. If it is not called 
    explicitly, it will be executed automatically, with an empty list of options,
    by the first call to initialize(). Do not call initialize_runtime() after 
    calling initialize().

    There is no corresponding terminate_runtime() call. The runtime is terminated
    automatically when the script or session ends.

    Parameters
        option_list - Python list of options; valid options are: 
                         -nodisplay (suppresses display functionality; Linux only)
                         -nojvm (disables the Java Virtual Machine)
    """
    if option_list:
        if not isinstance(option_list, list) and not isinstance(option_list, tuple):
            raise SyntaxError('initialize_runtime takes a list or tuple of strings.')
    _pir.initialize_runtime(option_list)

# terminate_runtime() is intentionally omitted. Instead, when running interactively, 
# the user should call exit(). When running a script, the runtime will automatically be
# terminated when the script ends.

@atexit.register
def __exit_packages():
    for package in _pir.instances_of_this_package:
        if package() is not None:
            package().terminate()
