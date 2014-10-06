'''Script for tidying up parallel STDOUT and STDERR files.

    This script scans the current directory and moves all STDOUT and STDERR files produced by a parallel run to the corresponding subdirectories.
'''
import os
import glob
import sys
import re
import shutil

def build_jobid_map(directory):
    '''Create a map from job ids to output directories.

        :arg path: Root directory
    '''
    jobid_map = {}
    contents = [subdir for subdir in glob.glob(os.path.join(directory,'*')) if os.path.isdir(subdir)]
    for subdir in contents:
        with open(os.path.join(subdir,'output.log')) as f:
            for line in f:
                m = re.match(' *PBS_JOBID *= *([0-9]+)\.sdb',line)
                if m:
                    jobid_map[int(m.group(1))] = subdir
    return jobid_map

def build_outputfile_map(directory):
    '''Create a map from job ids to STDOUT and STDERR files.

        :arg path: Root directory
    '''
    contents = [subdir for subdir in glob.glob(os.path.join(directory,'*')) if (not os.path.isdir(subdir))]
    outputfile_map = {}
    for filename in contents:
        m = re.match('.*\.[eo]([0-9]+)',filename)
        if m:
            jobid = int(m.group(1))
            if (not jobid in outputfile_map.keys()):
                outputfile_map[jobid] = []
            outputfile_map[jobid].append(filename)
    return outputfile_map

#################################################################
# M A I N
#################################################################
if (__name__ == '__main__'):
    if (len(sys.argv) > 2):
        print 'Usage: python '+sys.argv[0]+' [<directory>]'
        sys.exit(0)
    if (len(sys.argv)==2):
        directory = sys.argv[1]
    else:
        directory = '.'
    jobid_map = build_jobid_map(directory)
    outputfile_map = build_outputfile_map(directory)
    for jobid,subdir in jobid_map.iteritems():
        if jobid in outputfile_map.keys():
            for outputfile in outputfile_map[jobid]:
                print 'Moving '+os.path.basename(outputfile)+\
                      ' to '+os.path.basename(subdir)
                try:
                    shutil.move(outputfile,subdir)
                except shutil.Error as e:
                    print 'ERROR moving file: '+str(e)
