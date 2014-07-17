'''Trick for using the @profile decorator even without running the line
profiler.

This trick has been taken from stackoverflow, http://stackoverflow.com/questions/18229628/python-profiling-using-line-profiler-clever-way-to-remove-profile-statements
'''
import __builtin__

try:
    __builtin__.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    __builtin__.profile = profile
