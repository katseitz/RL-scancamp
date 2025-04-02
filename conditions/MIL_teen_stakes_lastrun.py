#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy2 Experiment Builder (v1.83.03), Mon Jun  4 14:09:35 2018
If you publish work using this script please cite the relevant PsychoPy publications
  Peirce, JW (2007) PsychoPy - Psychophysics software in Python. Journal of Neuroscience Methods, 162(1-2), 8-13.
  Peirce, JW (2009) Generating stimuli for neuroscience using PsychoPy. Frontiers in Neuroinformatics, 2:10. doi: 10.3389/neuro.11.010.2008
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
from psychopy import locale_setup, visual, core, data, event, logging, sound, gui
from psychopy.constants import *  # things like STARTED, FINISHED
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import sin, cos, tan, log, log10, pi, average, sqrt, std, deg2rad, rad2deg, linspace, asarray
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys # to get file system encoding

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__)).decode(sys.getfilesystemencoding())
os.chdir(_thisDir)

# Store info about the experiment session
expName = 'MIL_Stakes'  # from the Builder filename that created this script
expInfo = {u'session': u'001', u'mriMode': u'Off', u'vers': u'A', u'practice': u'yes', u'participant': u'CatP4_'}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False: core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + 'Subject_Data/%s_%s' %(expInfo['participant'], expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=u'/Users/catalystmri/Desktop/LearningTask/conditions/MIL_teen_stakes.psyexp',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
#save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.DEBUG)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(size=(1280, 800), fullscr=True, screen=0, allowGUI=True, allowStencil=False,
    monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb',
    blendMode='avg', useFBO=True,
    units='deg')
# store frame rate of monitor if we can measure it successfully
expInfo['frameRate']=win.getActualFrameRate()
if expInfo['frameRate']!=None:
    frameDur = 1.0/round(expInfo['frameRate'])
else:
    frameDur = 1.0/60.0 # couldn't get a reliable measure so guess

# Initialize components for Routine "pracDirections"
pracDirectionsClock = core.Clock()
PracDir = visual.TextStim(win=win, ori=0, name='PracDir',
    text='\n\nIn this game, your job is to find the correct picture. Sometimes, you might have to guess. During the game, the correct picture might change. Try to choose the picture that is correct most of the time.\n\nThere will be two pictures on the screen, one on the left and one on the right. Press the button with your pointer finger to choose the shape on the left, and press the button with your middle finger to choose the shape on the right. The shapes will change sides, but this does not affect whether or not the shape is correct.\n\nMake your choice as fast as you can. Once you choose, a box will show up on the screen. If you choose too late, your choice will not count.\n\nAfter you make a choice, you will see a + on the screen for a few seconds. Next, you will get feedback telling you if you are correct or incorrect.\n\n',    font='Arial',
    pos=[0, 0], height=0.75, wrapWidth=20,
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "pracDirections2"
pracDirections2Clock = core.Clock()
prac1image = visual.ImageStim(win=win, name='prac1image',units='pix', 
    image='sin', mask=None,
    ori=0, pos=[-180,0], size=265,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
prac2image = visual.ImageStim(win=win, name='prac2image',units='pix', 
    image='sin', mask=None,
    ori=0, pos=[180,0], size=265,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)

ChoiceDirs = visual.TextStim(win=win, ori=0, name='ChoiceDirs',
    text='There will be two pictures on the screen, one on the left and one on the right. Press the button with your pointer finger to choose the picture on the left, and press the button with your middle finger to choose the picture on the right. The pictures will change sides, but this does not affect whether or not the picture is correct.\n\n\n\n\n\n\n\n\n\n\n\nMake your choice as fast as you can. Once you choose, a box will show up on the screen. If you choose too late, your choice will not count.\n\nPress the button now to make a choice.\n',    font='Arial',
    pos=[0, 0], height=0.75, wrapWidth=25,
    color='white', colorSpace='rgb', opacity=1,
    depth=-5.0)

# Initialize components for Routine "pracDirections3"
pracDirections3Clock = core.Clock()
PracDirs3 = visual.TextStim(win=win, ori=0, name='PracDirs3',
    text='After you make a choice, you will see a + on the screen for a few seconds. Next, you will get feedback telling you if you are correct or incorrect.\n\nDo you have any questions?\n\nPress space to start the practice.\n',    font='Arial',
    pos=[0, 0], height=0.75, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "pracCue"
pracCueClock = core.Clock()
if expInfo['practice'] == "yes":
    pracOn=4
    pracBlockRun=1
if expInfo['practice'] == "no":
    pracOn=0
    pracBlockRun=0
#else:
#    pracOn=0
#    pracBlockRun=0



leftImage = visual.ImageStim(win=win, name='leftImage',units='pix', 
    image='sin', mask=None,
    ori=0, pos=[-180,0], size=265,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
rightimage = visual.ImageStim(win=win, name='rightimage',units='pix', 
    image='sin', mask=None,
    ori=0, pos=[180,0], size=265,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)


# Initialize components for Routine "pracJitter_2"
pracJitter_2Clock = core.Clock()
pracISI = visual.TextStim(win=win, ori=0, name='pracISI',
    text='+',    font='Arial',
    pos=[0, 0], height=1.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "pracOutcome"
pracOutcomeClock = core.Clock()
pracOutcome=None
pracOutcomeText = visual.TextStim(win=win, ori=0, name='pracOutcomeText',
    text='default text',    font='Arial',
    pos=[0, 0], height=1.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-1.0)
pracJitter = visual.TextStim(win=win, ori=0, name='pracJitter',
    text='+',    font='Arial',
    pos=[0, 0], height=1.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-2.0)

# Initialize components for Routine "gainDirs"
gainDirsClock = core.Clock()
if expInfo['vers']=="A":
    highgainColor="Tomato"
    lowgainColor="DarkTurquoise"
    highlossColor="Gold"
    lowlossColor="DarkOrchid"

if expInfo['vers']=="B":
    highgainColor="DarkTurquoise"
    lowgainColor="Gold"
    highlossColor="DarkOrchid"
    lowlossColor="Tomato"

if expInfo['vers']=="C":
    highgainColor="DarkOrchid"
    lowgainColor="Tomato"
    highlossColor="DarkTurquoise"
    lowlossColor="Gold"

if expInfo['vers']=="D":
    highgainColor="Gold"
    lowgainColor="DarkOrchid"
    highlossColor="Tomato"
    lowlossColor="DarkTurquoise"
highgainframe1_2 = visual.Rect(win=win, name='highgainframe1_2',
    width=[6, 5][0], height=[6, 5][1],
    ori=0, pos=[8, 3],
    lineWidth=1, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-2.0, 
interpolate=True)
highgainframe2_2 = visual.Rect(win=win, name='highgainframe2_2',
    width=[5, 4][0], height=[5, 4][1],
    ori=0, pos=[8, 3],
    lineWidth=1, lineColor=None, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-3.0, 
interpolate=True)
lowgainframe1_2 = visual.Rect(win=win, name='lowgainframe1_2',
    width=[6, 5][0], height=[6, 5][1],
    ori=0, pos=[8, -3],
    lineWidth=1, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-4.0, 
interpolate=True)
lowgainframe2_2 = visual.Rect(win=win, name='lowgainframe2_2',
    width=[5, 4][0], height=[5, 4][1],
    ori=0, pos=[8, -3],
    lineWidth=1, lineColor=None, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-5.0, 
interpolate=True)
highgaintoplabel_2 = visual.TextStim(win=win, ori=0, name='highgaintoplabel_2',
    text="+0.50",    font='Arial',
    pos=[8, 5.25], height=0.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-6.0)
highgainbottomlabel_2 = visual.TextStim(win=win, ori=0, name='highgainbottomlabel_2',
    text="+0.00",    font='Arial',
    pos=[8, 0.75], height=0.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-7.0)
lowgaintoplabel_2 = visual.TextStim(win=win, ori=0, name='lowgaintoplabel_2',
    text="+0.25",    font='Arial',
    pos=[8, -0.75], height=0.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-8.0)
lowgainbottomlabel_2 = visual.TextStim(win=win, ori=0, name='lowgainbottomlabel_2',
    text="+0.00",    font='Arial',
    pos=[8, -5.25], height=0.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-9.0)
frameDirText_2 = visual.TextStim(win=win, ori=0, name='frameDirText_2',
    text='Sometimes you can win money during the game!\n\nFor the HIGH WIN pair, you can win 50 cents if you are correct or win 0 cents if you are incorrect. \n\nFor the LOW WIN pair, you can win 25 cents if you are correct or win 0 cents if you are incorrect. \n\nThere will be a box around the pictures. The numbers in the box will tell you whether you can win a high or low amount. \n\nThe money you win will be paid to you as bonus at the end of the game, so try your best!',    font='Arial',
    pos=[-5, 0], height=0.8, wrapWidth=15,
    color='white', colorSpace='rgb', opacity=1,
    depth=-10.0)
highgainLabel_2 = visual.TextStim(win=win, ori=0, name='highgainLabel_2',
    text='HIGH WIN',    font='Arial',
    pos=[8, 3], height=0.8, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-11.0)
lowgainlabel_2 = visual.TextStim(win=win, ori=0, name='lowgainlabel_2',
    text='LOW WIN',    font='Arial',
    pos=[8, -3], height=0.8, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-12.0)

# Initialize components for Routine "lossDirs"
lossDirsClock = core.Clock()
highlossframe1 = visual.Rect(win=win, name='highlossframe1',
    width=[6, 5][0], height=[6, 5][1],
    ori=0, pos=[8, 3],
    lineWidth=1, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-1.0, 
interpolate=True)
highlossframe2 = visual.Rect(win=win, name='highlossframe2',
    width=[5, 4][0], height=[5, 4][1],
    ori=0, pos=[8, 3],
    lineWidth=1, lineColor=None, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-2.0, 
interpolate=True)
lowlossframe1 = visual.Rect(win=win, name='lowlossframe1',
    width=[6, 5][0], height=[6, 5][1],
    ori=0, pos=[8, -3],
    lineWidth=1, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-3.0, 
interpolate=True)
lowlossframe2 = visual.Rect(win=win, name='lowlossframe2',
    width=[5, 4][0], height=[5, 4][1],
    ori=0, pos=[8, -3],
    lineWidth=1, lineColor=None, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-4.0, 
interpolate=True)
highlosstoplabel = visual.TextStim(win=win, ori=0, name='highlosstoplabel',
    text="-0.00",    font='Arial',
    pos=[8, 5.25], height=0.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-5.0)
highlossbottomlabel = visual.TextStim(win=win, ori=0, name='highlossbottomlabel',
    text="-0.50",    font='Arial',
    pos=[8, 0.75], height=0.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-6.0)
lowlosstoplabel = visual.TextStim(win=win, ori=0, name='lowlosstoplabel',
    text="-0.00",    font='Arial',
    pos=[8, -0.75], height=0.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-7.0)
lowlossbottomlabel = visual.TextStim(win=win, ori=0, name='lowlossbottomlabel',
    text="-0.25",    font='Arial',
    pos=[8, -5.25], height=0.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-8.0)
frameDirText_3 = visual.TextStim(win=win, ori=0, name='frameDirText_3',
    text='Sometimes you can lose money during the game!\n\nFor the HIGH LOSE pair, you can lose 0 cents if you are correct or lose 50 cents if you are incorrect. \n\nFor the LOW LOSE pair, you can lose 0 cents if you are correct or lose 25 cents if you are incorrect. \n\nThere will be a box around the pictures. The The numbers in the box will tell you whether you can lose a high or low amount. \n\nThe money you lose will be taken away from the bonus money you win at the end of the game, so try your best!',    font='Arial',
    pos=[-5, 0], height=0.8, wrapWidth=15,
    color='white', colorSpace='rgb', opacity=1,
    depth=-9.0)
highlossLabel = visual.TextStim(win=win, ori=0, name='highlossLabel',
    text='HIGH LOSE',    font='Arial',
    pos=[8, 3], height=0.8, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-10.0)
lowlosslabel = visual.TextStim(win=win, ori=0, name='lowlosslabel',
    text='LOW LOSE',    font='Arial',
    pos=[8, -3], height=0.8, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-11.0)

# Initialize components for Routine "instructions"
instructionsClock = core.Clock()
instrMsg='''  

Directions to remember:  

1. Try to choose the picture that gives you the best chance of winning money and avoiding losing money. 

2. Press the left button with your POINTER finger to select the image on the left side of the screen. Press the right button with your MIDDLE finger to select the image on the right side of the screen.  

3. The pictures will sometimes appear on opposite sides of the screen. This does not change whether they will win or lose.  

4. Make your choice when you see the pictures. If you choose after that, your response won't be counted.  

5. The money that you win in this task will be YOURS TO KEEP.'''
instr_text = visual.TextStim(win=win, ori=0, name='instr_text',
    text=instrMsg,    font='Helvetica',
    pos=[0, 0], height=.8, wrapWidth=25,
    color='white', colorSpace='rgb', opacity=1,
    depth=-1.0)
response_keys = dict(left='1',right='2')
allowed_keys = response_keys.values()
milConditionsFile1 = os.path.join('conditions', 'MIL_stakes_cond%s_block1.csv' % expInfo['vers'])
milConditionsFile2 = os.path.join('conditions', 'MIL_stakes_cond%s_block2.csv' % expInfo['vers'])
milConditionsFile3 = os.path.join('conditions', 'MIL_stakes_cond%s_block3.csv' % expInfo['vers'])
milConditionsFile4 = os.path.join('conditions', 'MIL_stakes_cond%s_block4.csv' % expInfo['vers'])

logging.exp("Using conditions file: %s" % milConditionsFile1)

# Initialize components for Routine "waitForScanner"
waitForScannerClock = core.Clock()
fmriClock = core.Clock()
trigger = 'usb'
#trigger = 'parallel'
if trigger == 'parallel':
    from psychopy.contrib import parallel as winioport
    #from psychopy import parallel
elif trigger == 'usb':
    from psychopy.hardware.emulator import launchScan
    #
    # settings for launchScan:
    MR_settings = { 
        'TR': 2, # duration (sec) per volume
        'volumes': 210, # number of whole-brain 3D volumes / frames
        'sync': 'equal', # character to use as the sync timing event; assumed to come at start of a volume
        'skip': 0, # number of volumes lacking a sync pulse at start of scan (for T1 stabilization)
        }


# Initialize components for Routine "prepRun"
prepRunClock = core.Clock()
trialOrder=None
startFix1 = visual.TextStim(win=win, ori=0, name='startFix1',
    text='+',    font='Arial',
    pos=[0, 0], height=1.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-1.0)

# Initialize components for Routine "cue"
cueClock = core.Clock()
topLabel=None
bottomLabel=None
framecolor=None


frame1 = visual.Rect(win=win, name='frame1',
    width=[18, 10][0], height=[18, 10][1],
    ori=0, pos=[0, 0],
    lineWidth=2, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-1.0, 
interpolate=True)
frame2 = visual.Rect(win=win, name='frame2',
    width=[16, 7.5][0], height=[16, 7.5][1],
    ori=0, pos=[0, 0],
    lineWidth=2, lineColor=None, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-2.0, 
interpolate=True)
import json
jsonfile = os.path.join('Subject_Data/', expInfo['participant'] + '.json')
# Assign Cues to Condition or load previous set.
if os.path.exists(jsonfile):
    with open(jsonfile,'r') as f:
        cuePairs = json.loads(f.read())
else:

    imageList = os.path.join('conditions/', 'PicList_%s.csv' % expInfo['vers']) 
    with open(imageList, 'r') as f:
        allImages=[s.strip() for s in f.readlines()]
        cuePairs = {
            'highgain': {'optimalChoice': allImages[0], 'suboptimalChoice': allImages[1]}, 
            'lowgain': {'optimalChoice': allImages[2], 'suboptimalChoice': allImages[3]}, 
            'highloss': {'optimalChoice': allImages[4], 'suboptimalChoice': allImages[5]},
            'lowloss': {'optimalChoice': allImages[6], 'suboptimalChoice': allImages[7]}
    }

    logging.info('Cue Pairs: %s' % cuePairs)
    expInfo['cuePairs'] = cuePairs
    with open(jsonfile, 'w') as f:
        f.write(json.dumps(cuePairs, sort_keys=True, indent=4))


leftCue = visual.PatchStim(win=win, name='leftCue',units='pix', 
    tex='sin', mask=None,
    ori=0, pos=[-180, 0], size=276, sf=None, phase=0.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    texRes=128, interpolate=True, depth=-5.0)
rightCue = visual.PatchStim(win=win, name='rightCue',units='pix', 
    tex='sin', mask=None,
    ori=0, pos=[180, 0], size=276, sf=None, phase=0.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    texRes=128, interpolate=True, depth=-6.0)
import time
expInfo['expStartTime'] = time.ctime()

TRIAL_DURATION = 10
trialsClock = core.Clock()

topFrameText = visual.TextStim(win=win, ori=0, name='topFrameText',
    text='default text',    font='Arial',
    pos=[0, 4.5], height=1, wrapWidth=None,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    depth=-9.0)
bottomFrameText = visual.TextStim(win=win, ori=0, name='bottomFrameText',
    text='default text',    font='Arial',
    pos=[0, -4.25], height=1, wrapWidth=None,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    depth=-10.0)
selectionDuration = 2.5

def selectionPosition(response):
    '''Set position of selectionCue'''
    side = 1 if response == response_keys['right'] else -1  # For fMRI, button 1 == left and button 2 == right, so positive will shift right and negative will shift left
    return [side * 180,0]

# Cue Choice Indicator
selectionIndicator = visual.Polygon(win, edges =4, ori=45, radius=195, 
    name = 'selectionIndicator', lineColor = 'white', units ='pix', lineWidth=2, interpolate=True)

#vertices=[[0,0], [0,260], [260,260], [260,0]]

Choice=None

# Initialize components for Routine "outcomeDelay"
outcomeDelayClock = core.Clock()
frame1_fix = visual.Rect(win=win, name='frame1_fix',
    width=[18, 10][0], height=[18, 10][1],
    ori=0, pos=[0, 0],
    lineWidth=2, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=0.0, 
interpolate=True)
frame2_fix = visual.Rect(win=win, name='frame2_fix',
    width=[16, 7.5][0], height=[16, 7.5][1],
    ori=0, pos=[0, 0],
    lineWidth=2, lineColor=None, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-1.0, 
interpolate=True)
outcomeDelayFix = visual.TextStim(win=win, ori=0, name='outcomeDelayFix',
    text='+',    font='Arial',
    pos=[0, 0], height=1.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=-2.0)
topFrameText_fix = visual.TextStim(win=win, ori=0, name='topFrameText_fix',
    text='default text',    font='Arial',
    pos=[0, 4.5], height=1, wrapWidth=None,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    depth=-3.0)
bottomFrameText_fix = visual.TextStim(win=win, ori=0, name='bottomFrameText_fix',
    text='default text',    font='Arial',
    pos=[0, -4.25], height=1, wrapWidth=None,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    depth=-4.0)

# Initialize components for Routine "outcome"
outcomeClock = core.Clock()
Accuracy=None
bank = 0
def choose_outcome_from_cue(response):
    '''Choose an outcome to display (if True, then the trial action will occur, if False it won't) 
    by setting a high or low threshold based on the response and image chosen, and then
    see if a random "roll" exceeds the threshold.

     The roll must be greater than threshold to return True, so a high probability will have a low
     threshold, and vice versa.'''
       
    if sidePositions[optimalImg] == cueResp.keys:
        return True if (optcorrect==1) else False
    elif sidePositions[suboptimalImg] == cueResp.keys:
        return True if (optcorrect==0) else False

Accuracy_OptChoice=None


frame1_fb = visual.Rect(win=win, name='frame1_fb',
    width=[18, 10][0], height=[18, 10][1],
    ori=0, pos=[0, 0],
    lineWidth=2, lineColor=1.0, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-2.0, 
interpolate=True)
frame2_fb = visual.Rect(win=win, name='frame2_fb',
    width=[16, 7.5][0], height=[16, 7.5][1],
    ori=0, pos=[0, 0],
    lineWidth=2, lineColor=None, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-3.0, 
interpolate=True)
outcome_text = visual.TextStim(win=win, ori=0, name='outcome_text',
    text='default text',    font='Arial',
    pos=[0, 0], height=1.5, wrapWidth=20,
    color='white', colorSpace='rgb', opacity=1,
    depth=-4.0)
topFrameText_FB = visual.TextStim(win=win, ori=0, name='topFrameText_FB',
    text='default text',    font='Arial',
    pos=[0, 4.5], height=1, wrapWidth=None,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    depth=-5.0)
bottomFrameText_FB = visual.TextStim(win=win, ori=0, name='bottomFrameText_FB',
    text='default text',    font='Arial',
    pos=[0, -4.25], height=1, wrapWidth=None,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    depth=-6.0)

# Initialize components for Routine "fixation"
fixationClock = core.Clock()
fixation_crosshair = visual.TextStim(win=win, ori=0, name='fixation_crosshair',
    text='+',    font='Arial',
    pos=[0, 0], height=1.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0)


fixFrame1 = visual.Rect(win=win, name='fixFrame1',
    width=[18, 10][0], height=[18, 10][1],
    ori=0, pos=[0, 0],
    lineWidth=2, lineColor=[0,0,0], lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-3.0, 
interpolate=True)
fixFrame2 = visual.Rect(win=win, name='fixFrame2',
    width=[16, 7.5][0], height=[16, 7.5][1],
    ori=0, pos=[0, 0],
    lineWidth=2, lineColor=None, lineColorSpace='rgb',
    fillColor=None, fillColorSpace='rgb',
    opacity=1,depth=-4.0, 
interpolate=True)

# Initialize components for Routine "breakscreen"
breakscreenClock = core.Clock()
breakText = visual.TextStim(win=win, ori=0, name='breakText',
    text='You finished this round!\n\nPress space to continue playing.',    font='Arial',
    pos=[0, 0], height=1, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "ratingDir"
ratingDirClock = core.Clock()
RatingDirs = visual.TextStim(win=win, ori=0, name='RatingDirs',
    text='You finished the game!\n\nNext, you will see a series of pictures. Some will be from the game, and some will be unfamiliar.\n\nPlease choose the outcome that was most often associated with the picture. If the picture is unfamiliar, please select the N/A option.  \n\nNext, please rate how much you like the image and how intense that feeling is.\n\nPress space to start.',    font='Arial',
    pos=[0, 0], height=1, wrapWidth=20,
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "rating"
ratingClock = core.Clock()
image = visual.ImageStim(win=win, name='image',units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0, 0], size=275,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
pic_rating = visual.RatingScale(win=win, name='pic_rating', marker='triangle', size=1.0, pos=[0.0, -0.4], choices=[u'-$0.50', u'-$0.25', u'$0', u'+$0.25', u'+$0.50', u'N/A'], tickHeight=-1)
Question = visual.TextStim(win=win, ori=0, name='Question',
    text='Which outcome was associated with this picture most of the time?',    font='Arial',
    pos=[0, 4], height=0.8, wrapWidth=20,
    color='white', colorSpace='rgb', opacity=1,
    depth=-2.0)

# Initialize components for Routine "ratingValence"
ratingValenceClock = core.Clock()
image_2 = visual.ImageStim(win=win, name='image_2',units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0, 0], size=275,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
pic_rating_2 = visual.RatingScale(win=win, name='pic_rating_2', marker='triangle', size=1.0, pos=[0.0, -0.4], low=-50, high=50, labels=['very bad', ' very good'], scale='')
Question_2 = visual.TextStim(win=win, ori=0, name='Question_2',
    text='How does this picture make you feel?',    font='Arial',
    pos=[0, 4], height=0.8, wrapWidth=20,
    color='white', colorSpace='rgb', opacity=1,
    depth=-2.0)

# Initialize components for Routine "ratingArousal"
ratingArousalClock = core.Clock()
image_3 = visual.ImageStim(win=win, name='image_3',units='pix', 
    image='sin', mask=None,
    ori=0, pos=[0, 0], size=275,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)
pic_rating_3 = visual.RatingScale(win=win, name='pic_rating_3', marker='triangle', size=1.0, pos=[0.0, -0.4], low=0, high=100, labels=['not strong', ' very strong'], scale='')
Question_3 = visual.TextStim(win=win, ori=0, name='Question_3',
    text='How strong is your feeling about this picture?',    font='Arial',
    pos=[0, 4], height=0.8, wrapWidth=20,
    color='white', colorSpace='rgb', opacity=1,
    depth=-2.0)

# Initialize components for Routine "Done"
DoneClock = core.Clock()
EndExperiment = visual.TextStim(win=win, ori=0, name='EndExperiment',
    text='default text',    font='Arial',
    pos=[0, 0], height=1.5, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0)
#end_msg=None

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# set up handler to look after randomisation of conditions etc
pracBlock = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='pracBlock')
thisExp.addLoop(pracBlock)  # add the loop to the experiment
thisPracBlock = pracBlock.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb=thisPracBlock.rgb)
if thisPracBlock != None:
    for paramName in thisPracBlock.keys():
        exec(paramName + '= thisPracBlock.' + paramName)

for thisPracBlock in pracBlock:
    currentLoop = pracBlock
    # abbreviate parameter names if possible (e.g. rgb = thisPracBlock.rgb)
    if thisPracBlock != None:
        for paramName in thisPracBlock.keys():
            exec(paramName + '= thisPracBlock.' + paramName)
    
    #------Prepare to start Routine "pracDirections"-------
    t = 0
    pracDirectionsClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    startPrac = event.BuilderKeyResponse()  # create an object of type KeyResponse
    startPrac.status = NOT_STARTED
    # keep track of which components have finished
    pracDirectionsComponents = []
    pracDirectionsComponents.append(PracDir)
    pracDirectionsComponents.append(startPrac)
    for thisComponent in pracDirectionsComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    #-------Start Routine "pracDirections"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = pracDirectionsClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *PracDir* updates
        if t >= 0.0 and PracDir.status == NOT_STARTED:
            # keep track of start time/frame for later
            PracDir.tStart = t  # underestimates by a little under one frame
            PracDir.frameNStart = frameN  # exact frame index
            PracDir.setAutoDraw(True)
        
        # *startPrac* updates
        if t >= 0.0 and startPrac.status == NOT_STARTED:
            # keep track of start time/frame for later
            startPrac.tStart = t  # underestimates by a little under one frame
            startPrac.frameNStart = frameN  # exact frame index
            startPrac.status = STARTED
            # keyboard checking is just starting
            event.clearEvents(eventType='keyboard')
        if startPrac.status == STARTED:
            theseKeys = event.getKeys(keyList=['space'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                # a response ends the routine
                continueRoutine = False
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pracDirectionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    #-------Ending Routine "pracDirections"-------
    for thisComponent in pracDirectionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "pracDirections" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    #------Prepare to start Routine "pracDirections2"-------
    t = 0
    pracDirections2Clock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    prac1image.setImage('images/prac1.png')
    prac2image.setImage('images/prac2.png')
    pracResp_dirs = event.BuilderKeyResponse()  # create an object of type KeyResponse
    pracResp_dirs.status = NOT_STARTED
    cueEndTime = 20
    selectionIndicator.status = NOT_STARTED
    advanceDirs = event.BuilderKeyResponse()  # create an object of type KeyResponse
    advanceDirs.status = NOT_STARTED
    # keep track of which components have finished
    pracDirections2Components = []
    pracDirections2Components.append(prac1image)
    pracDirections2Components.append(prac2image)
    pracDirections2Components.append(pracResp_dirs)
    pracDirections2Components.append(advanceDirs)
    pracDirections2Components.append(ChoiceDirs)
    for thisComponent in pracDirections2Components:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    #-------Start Routine "pracDirections2"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = pracDirections2Clock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *prac1image* updates
        if t >= 0.0 and prac1image.status == NOT_STARTED:
            # keep track of start time/frame for later
            prac1image.tStart = t  # underestimates by a little under one frame
            prac1image.frameNStart = frameN  # exact frame index
            prac1image.setAutoDraw(True)
        
        # *prac2image* updates
        if t >= 0.0 and prac2image.status == NOT_STARTED:
            # keep track of start time/frame for later
            prac2image.tStart = t  # underestimates by a little under one frame
            prac2image.frameNStart = frameN  # exact frame index
            prac2image.setAutoDraw(True)
        
        # *pracResp_dirs* updates
        if t >= 0.0 and pracResp_dirs.status == NOT_STARTED:
            # keep track of start time/frame for later
            pracResp_dirs.tStart = t  # underestimates by a little under one frame
            pracResp_dirs.frameNStart = frameN  # exact frame index
            pracResp_dirs.status = STARTED
            # keyboard checking is just starting
            win.callOnFlip(pracResp_dirs.clock.reset)  # t=0 on next screen flip
            event.clearEvents(eventType='keyboard')
        if pracResp_dirs.status == STARTED and t >= (0.0 + (2.5-win.monitorFramePeriod*0.75)): #most of one frame period left
            pracResp_dirs.status = STOPPED
        if pracResp_dirs.status == STARTED:
            theseKeys = event.getKeys(keyList=['1', '2'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                if pracResp_dirs.keys == []:  # then this was the first keypress
                    pracResp_dirs.keys = theseKeys[0]  # just the first key pressed
                    pracResp_dirs.rt = pracResp_dirs.clock.getTime()
        # If the selection has been drawn for the selection duration,
        # or the trial should end (cutting off the selection duration), end the routine
        if t > cueEndTime:
            continueRoutine = False
        
        # Start Drawing Selection Indicator if Selection has been made
        if pracResp_dirs.keys and selectionIndicator.status == NOT_STARTED:
            pracDirections2Components.append(selectionIndicator)
            selectionIndicator.status = STARTED
            selectionIndicator.setPos(selectionPosition(pracResp_dirs.keys))
            selectionIndicator.setAutoDraw(True)
            #if getSnapshots: win.getMovieFrame()
            # Set End Time, clip to end of trial length (4 sec max)
            # Don't worry about end of selection display in fMRI
            #cueEndTime = t + selectionDuration
        
        
        # *advanceDirs* updates
        if t >= 0.0 and advanceDirs.status == NOT_STARTED:
            # keep track of start time/frame for later
            advanceDirs.tStart = t  # underestimates by a little under one frame
            advanceDirs.frameNStart = frameN  # exact frame index
            advanceDirs.status = STARTED
            # keyboard checking is just starting
            event.clearEvents(eventType='keyboard')
        if advanceDirs.status == STARTED:
            theseKeys = event.getKeys(keyList=['space'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                # a response ends the routine
                continueRoutine = False
        
        # *ChoiceDirs* updates
        if t >= 0.0 and ChoiceDirs.status == NOT_STARTED:
            # keep track of start time/frame for later
            ChoiceDirs.tStart = t  # underestimates by a little under one frame
            ChoiceDirs.frameNStart = frameN  # exact frame index
            ChoiceDirs.setAutoDraw(True)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pracDirections2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    #-------Ending Routine "pracDirections2"-------
    for thisComponent in pracDirections2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if pracResp_dirs.keys in ['', [], None]:  # No response was made
       pracResp_dirs.keys=None
    # store data for pracBlock (TrialHandler)
    pracBlock.addData('pracResp_dirs.keys',pracResp_dirs.keys)
    if pracResp_dirs.keys != None:  # we had a response
        pracBlock.addData('pracResp_dirs.rt', pracResp_dirs.rt)
    
    # the Routine "pracDirections2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practiceBlock = data.TrialHandler(nReps=pracBlockRun, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='practiceBlock')
    thisExp.addLoop(practiceBlock)  # add the loop to the experiment
    thisPracticeBlock = practiceBlock.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb=thisPracticeBlock.rgb)
    if thisPracticeBlock != None:
        for paramName in thisPracticeBlock.keys():
            exec(paramName + '= thisPracticeBlock.' + paramName)
    
    for thisPracticeBlock in practiceBlock:
        currentLoop = practiceBlock
        # abbreviate parameter names if possible (e.g. rgb = thisPracticeBlock.rgb)
        if thisPracticeBlock != None:
            for paramName in thisPracticeBlock.keys():
                exec(paramName + '= thisPracticeBlock.' + paramName)
        
        #------Prepare to start Routine "pracDirections3"-------
        t = 0
        pracDirections3Clock.reset()  # clock 
        frameN = -1
        # update component parameters for each repeat
        advScrn = event.BuilderKeyResponse()  # create an object of type KeyResponse
        advScrn.status = NOT_STARTED
        # keep track of which components have finished
        pracDirections3Components = []
        pracDirections3Components.append(PracDirs3)
        pracDirections3Components.append(advScrn)
        for thisComponent in pracDirections3Components:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        
        #-------Start Routine "pracDirections3"-------
        continueRoutine = True
        while continueRoutine:
            # get current time
            t = pracDirections3Clock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *PracDirs3* updates
            if t >= 0.0 and PracDirs3.status == NOT_STARTED:
                # keep track of start time/frame for later
                PracDirs3.tStart = t  # underestimates by a little under one frame
                PracDirs3.frameNStart = frameN  # exact frame index
                PracDirs3.setAutoDraw(True)
            
            # *advScrn* updates
            if t >= 0.0 and advScrn.status == NOT_STARTED:
                # keep track of start time/frame for later
                advScrn.tStart = t  # underestimates by a little under one frame
                advScrn.frameNStart = frameN  # exact frame index
                advScrn.status = STARTED
                # keyboard checking is just starting
                event.clearEvents(eventType='keyboard')
            if advScrn.status == STARTED:
                theseKeys = event.getKeys(keyList=['space'])
                
                # check for quit:
                if "escape" in theseKeys:
                    endExpNow = True
                if len(theseKeys) > 0:  # at least one key was pressed
                    # a response ends the routine
                    continueRoutine = False
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pracDirections3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                core.quit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        #-------Ending Routine "pracDirections3"-------
        for thisComponent in pracDirections3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # the Routine "pracDirections3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        practice = data.TrialHandler(nReps=pracOn, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('conditions/MIL_teen_prac.xlsx'),
            seed=None, name='practice')
        thisExp.addLoop(practice)  # add the loop to the experiment
        thisPractice = practice.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb=thisPractice.rgb)
        if thisPractice != None:
            for paramName in thisPractice.keys():
                exec(paramName + '= thisPractice.' + paramName)
        
        for thisPractice in practice:
            currentLoop = practice
            # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
            if thisPractice != None:
                for paramName in thisPractice.keys():
                    exec(paramName + '= thisPractice.' + paramName)
            
            #------Prepare to start Routine "pracCue"-------
            t = 0
            pracCueClock.reset()  # clock 
            frameN = -1
            routineTimer.add(2.500000)
            # update component parameters for each repeat
            
            leftImage.setImage(LeftImage)
            rightimage.setImage(RightImage)
            pracResp = event.BuilderKeyResponse()  # create an object of type KeyResponse
            pracResp.status = NOT_STARTED
            cueEndTime = 2.5
            selectionIndicator.status = NOT_STARTED
            # keep track of which components have finished
            pracCueComponents = []
            pracCueComponents.append(leftImage)
            pracCueComponents.append(rightimage)
            pracCueComponents.append(pracResp)
            for thisComponent in pracCueComponents:
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            
            #-------Start Routine "pracCue"-------
            continueRoutine = True
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = pracCueClock.getTime()
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                
                # *leftImage* updates
                if t >= 0.0 and leftImage.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    leftImage.tStart = t  # underestimates by a little under one frame
                    leftImage.frameNStart = frameN  # exact frame index
                    leftImage.setAutoDraw(True)
                if leftImage.status == STARTED and t >= (0.0 + (2.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                    leftImage.setAutoDraw(False)
                
                # *rightimage* updates
                if t >= 0.0 and rightimage.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    rightimage.tStart = t  # underestimates by a little under one frame
                    rightimage.frameNStart = frameN  # exact frame index
                    rightimage.setAutoDraw(True)
                if rightimage.status == STARTED and t >= (0.0 + (2.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                    rightimage.setAutoDraw(False)
                
                # *pracResp* updates
                if t >= 0.0 and pracResp.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    pracResp.tStart = t  # underestimates by a little under one frame
                    pracResp.frameNStart = frameN  # exact frame index
                    pracResp.status = STARTED
                    # keyboard checking is just starting
                    win.callOnFlip(pracResp.clock.reset)  # t=0 on next screen flip
                    event.clearEvents(eventType='keyboard')
                if pracResp.status == STARTED and t >= (0.0 + (2.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                    pracResp.status = STOPPED
                if pracResp.status == STARTED:
                    theseKeys = event.getKeys(keyList=['1', '2'])
                    
                    # check for quit:
                    if "escape" in theseKeys:
                        endExpNow = True
                    if len(theseKeys) > 0:  # at least one key was pressed
                        if pracResp.keys == []:  # then this was the first keypress
                            pracResp.keys = theseKeys[0]  # just the first key pressed
                            pracResp.rt = pracResp.clock.getTime()
                            # was this 'correct'?
                            if (pracResp.keys == str(PracCorrect)) or (pracResp.keys == PracCorrect):
                                pracResp.corr = 1
                            else:
                                pracResp.corr = 0
                # If the selection has been drawn for the selection duration,
                # or the trial should end (cutting off the selection duration), end the routine
                if t > cueEndTime:
                    continueRoutine = False
                
                # Start Drawing Selection Indicator if Selection has been made
                if pracResp.keys and selectionIndicator.status == NOT_STARTED:
                    pracCueComponents.append(selectionIndicator)
                    selectionIndicator.status = STARTED
                    selectionIndicator.setPos(selectionPosition(pracResp.keys))
                    selectionIndicator.setAutoDraw(True)
                    #if getSnapshots: win.getMovieFrame()
                    # Set End Time, clip to end of trial length (4 sec max)
                    # Don't worry about end of selection display in fMRI
                    #cueEndTime = t + selectionDuration
                
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in pracCueComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # check for quit (the Esc key)
                if endExpNow or event.getKeys(keyList=["escape"]):
                    core.quit()
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            #-------Ending Routine "pracCue"-------
            for thisComponent in pracCueComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            
            # check responses
            if pracResp.keys in ['', [], None]:  # No response was made
               pracResp.keys=None
               # was no response the correct answer?!
               if str(PracCorrect).lower() == 'none': pracResp.corr = 1  # correct non-response
               else: pracResp.corr = 0  # failed to respond (incorrectly)
            # store data for practice (TrialHandler)
            practice.addData('pracResp.keys',pracResp.keys)
            practice.addData('pracResp.corr', pracResp.corr)
            if pracResp.keys != None:  # we had a response
                practice.addData('pracResp.rt', pracResp.rt)
            
            
            #------Prepare to start Routine "pracJitter_2"-------
            t = 0
            pracJitter_2Clock.reset()  # clock 
            frameN = -1
            routineTimer.add(2.500000)
            # update component parameters for each repeat
            # keep track of which components have finished
            pracJitter_2Components = []
            pracJitter_2Components.append(pracISI)
            for thisComponent in pracJitter_2Components:
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            
            #-------Start Routine "pracJitter_2"-------
            continueRoutine = True
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = pracJitter_2Clock.getTime()
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *pracISI* updates
                if t >= 0.0 and pracISI.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    pracISI.tStart = t  # underestimates by a little under one frame
                    pracISI.frameNStart = frameN  # exact frame index
                    pracISI.setAutoDraw(True)
                if pracISI.status == STARTED and t >= (0.0 + (2.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                    pracISI.setAutoDraw(False)
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in pracJitter_2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # check for quit (the Esc key)
                if endExpNow or event.getKeys(keyList=["escape"]):
                    core.quit()
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            #-------Ending Routine "pracJitter_2"-------
            for thisComponent in pracJitter_2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            
            #------Prepare to start Routine "pracOutcome"-------
            t = 0
            pracOutcomeClock.reset()  # clock 
            frameN = -1
            routineTimer.add(2.500000)
            # update component parameters for each repeat
            if pracResp.keys:
                if pracResp.corr == 1:
                    pracOutcome="Correct!"
                elif pracResp.corr == 0:
                    pracOutcome= "Incorrect!"
            else: pracOutcome= "Missed"
            pracOutcomeText.setText(pracOutcome)
            # keep track of which components have finished
            pracOutcomeComponents = []
            pracOutcomeComponents.append(pracOutcomeText)
            pracOutcomeComponents.append(pracJitter)
            for thisComponent in pracOutcomeComponents:
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            
            #-------Start Routine "pracOutcome"-------
            continueRoutine = True
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = pracOutcomeClock.getTime()
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                
                # *pracOutcomeText* updates
                if t >= 0.0 and pracOutcomeText.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    pracOutcomeText.tStart = t  # underestimates by a little under one frame
                    pracOutcomeText.frameNStart = frameN  # exact frame index
                    pracOutcomeText.setAutoDraw(True)
                if pracOutcomeText.status == STARTED and t >= (0.0 + (1.0-win.monitorFramePeriod*0.75)): #most of one frame period left
                    pracOutcomeText.setAutoDraw(False)
                
                # *pracJitter* updates
                if t >= 1.0 and pracJitter.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    pracJitter.tStart = t  # underestimates by a little under one frame
                    pracJitter.frameNStart = frameN  # exact frame index
                    pracJitter.setAutoDraw(True)
                if pracJitter.status == STARTED and t >= (1.0 + (1.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                    pracJitter.setAutoDraw(False)
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in pracOutcomeComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # check for quit (the Esc key)
                if endExpNow or event.getKeys(keyList=["escape"]):
                    core.quit()
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            #-------Ending Routine "pracOutcome"-------
            for thisComponent in pracOutcomeComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            
            thisExp.nextEntry()
            
        # completed pracOn repeats of 'practice'
        
        # get names of stimulus parameters
        if practice.trialList in ([], [None], None):  params = []
        else:  params = practice.trialList[0].keys()
        # save data for this loop
        practice.saveAsExcel(filename + '.xlsx', sheetName='practice',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
    # completed pracBlockRun repeats of 'practiceBlock'
    
    
    #------Prepare to start Routine "gainDirs"-------
    t = 0
    gainDirsClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    advanceScrn_2 = event.BuilderKeyResponse()  # create an object of type KeyResponse
    advanceScrn_2.status = NOT_STARTED
    
    highgainframe1_2.setLineColor([0,0,0])
    lowgainframe1_2.setLineColor([0,0,0])
    # keep track of which components have finished
    gainDirsComponents = []
    gainDirsComponents.append(advanceScrn_2)
    gainDirsComponents.append(highgainframe1_2)
    gainDirsComponents.append(highgainframe2_2)
    gainDirsComponents.append(lowgainframe1_2)
    gainDirsComponents.append(lowgainframe2_2)
    gainDirsComponents.append(highgaintoplabel_2)
    gainDirsComponents.append(highgainbottomlabel_2)
    gainDirsComponents.append(lowgaintoplabel_2)
    gainDirsComponents.append(lowgainbottomlabel_2)
    gainDirsComponents.append(frameDirText_2)
    gainDirsComponents.append(highgainLabel_2)
    gainDirsComponents.append(lowgainlabel_2)
    for thisComponent in gainDirsComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    #-------Start Routine "gainDirs"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = gainDirsClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *advanceScrn_2* updates
        if t >= 0.0 and advanceScrn_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            advanceScrn_2.tStart = t  # underestimates by a little under one frame
            advanceScrn_2.frameNStart = frameN  # exact frame index
            advanceScrn_2.status = STARTED
            # keyboard checking is just starting
            event.clearEvents(eventType='keyboard')
        if advanceScrn_2.status == STARTED:
            theseKeys = event.getKeys(keyList=['space', 'a'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                # a response ends the routine
                continueRoutine = False
        
        
        # *highgainframe1_2* updates
        if t >= 0.0 and highgainframe1_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            highgainframe1_2.tStart = t  # underestimates by a little under one frame
            highgainframe1_2.frameNStart = frameN  # exact frame index
            highgainframe1_2.setAutoDraw(True)
        
        # *highgainframe2_2* updates
        if t >= 0.0 and highgainframe2_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            highgainframe2_2.tStart = t  # underestimates by a little under one frame
            highgainframe2_2.frameNStart = frameN  # exact frame index
            highgainframe2_2.setAutoDraw(True)
        
        # *lowgainframe1_2* updates
        if t >= 0.0 and lowgainframe1_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            lowgainframe1_2.tStart = t  # underestimates by a little under one frame
            lowgainframe1_2.frameNStart = frameN  # exact frame index
            lowgainframe1_2.setAutoDraw(True)
        
        # *lowgainframe2_2* updates
        if t >= 0.0 and lowgainframe2_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            lowgainframe2_2.tStart = t  # underestimates by a little under one frame
            lowgainframe2_2.frameNStart = frameN  # exact frame index
            lowgainframe2_2.setAutoDraw(True)
        
        # *highgaintoplabel_2* updates
        if t >= 0.0 and highgaintoplabel_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            highgaintoplabel_2.tStart = t  # underestimates by a little under one frame
            highgaintoplabel_2.frameNStart = frameN  # exact frame index
            highgaintoplabel_2.setAutoDraw(True)
        
        # *highgainbottomlabel_2* updates
        if t >= 0.0 and highgainbottomlabel_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            highgainbottomlabel_2.tStart = t  # underestimates by a little under one frame
            highgainbottomlabel_2.frameNStart = frameN  # exact frame index
            highgainbottomlabel_2.setAutoDraw(True)
        
        # *lowgaintoplabel_2* updates
        if t >= 0.0 and lowgaintoplabel_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            lowgaintoplabel_2.tStart = t  # underestimates by a little under one frame
            lowgaintoplabel_2.frameNStart = frameN  # exact frame index
            lowgaintoplabel_2.setAutoDraw(True)
        
        # *lowgainbottomlabel_2* updates
        if t >= 0.0 and lowgainbottomlabel_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            lowgainbottomlabel_2.tStart = t  # underestimates by a little under one frame
            lowgainbottomlabel_2.frameNStart = frameN  # exact frame index
            lowgainbottomlabel_2.setAutoDraw(True)
        
        # *frameDirText_2* updates
        if t >= 0.0 and frameDirText_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            frameDirText_2.tStart = t  # underestimates by a little under one frame
            frameDirText_2.frameNStart = frameN  # exact frame index
            frameDirText_2.setAutoDraw(True)
        
        # *highgainLabel_2* updates
        if t >= 0.0 and highgainLabel_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            highgainLabel_2.tStart = t  # underestimates by a little under one frame
            highgainLabel_2.frameNStart = frameN  # exact frame index
            highgainLabel_2.setAutoDraw(True)
        
        # *lowgainlabel_2* updates
        if t >= 0.0 and lowgainlabel_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            lowgainlabel_2.tStart = t  # underestimates by a little under one frame
            lowgainlabel_2.frameNStart = frameN  # exact frame index
            lowgainlabel_2.setAutoDraw(True)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in gainDirsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    #-------Ending Routine "gainDirs"-------
    for thisComponent in gainDirsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    
    # the Routine "gainDirs" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    #------Prepare to start Routine "lossDirs"-------
    t = 0
    lossDirsClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    advanceScrn_3 = event.BuilderKeyResponse()  # create an object of type KeyResponse
    advanceScrn_3.status = NOT_STARTED
    highlossframe1.setLineColor([0,0,0])
    lowlossframe1.setLineColor([0,0,0])
    # keep track of which components have finished
    lossDirsComponents = []
    lossDirsComponents.append(advanceScrn_3)
    lossDirsComponents.append(highlossframe1)
    lossDirsComponents.append(highlossframe2)
    lossDirsComponents.append(lowlossframe1)
    lossDirsComponents.append(lowlossframe2)
    lossDirsComponents.append(highlosstoplabel)
    lossDirsComponents.append(highlossbottomlabel)
    lossDirsComponents.append(lowlosstoplabel)
    lossDirsComponents.append(lowlossbottomlabel)
    lossDirsComponents.append(frameDirText_3)
    lossDirsComponents.append(highlossLabel)
    lossDirsComponents.append(lowlosslabel)
    for thisComponent in lossDirsComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    #-------Start Routine "lossDirs"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = lossDirsClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *advanceScrn_3* updates
        if t >= 0.0 and advanceScrn_3.status == NOT_STARTED:
            # keep track of start time/frame for later
            advanceScrn_3.tStart = t  # underestimates by a little under one frame
            advanceScrn_3.frameNStart = frameN  # exact frame index
            advanceScrn_3.status = STARTED
            # keyboard checking is just starting
            event.clearEvents(eventType='keyboard')
        if advanceScrn_3.status == STARTED:
            theseKeys = event.getKeys(keyList=['space', 'a'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                # a response ends the routine
                continueRoutine = False
        
        # *highlossframe1* updates
        if t >= 0.0 and highlossframe1.status == NOT_STARTED:
            # keep track of start time/frame for later
            highlossframe1.tStart = t  # underestimates by a little under one frame
            highlossframe1.frameNStart = frameN  # exact frame index
            highlossframe1.setAutoDraw(True)
        
        # *highlossframe2* updates
        if t >= 0.0 and highlossframe2.status == NOT_STARTED:
            # keep track of start time/frame for later
            highlossframe2.tStart = t  # underestimates by a little under one frame
            highlossframe2.frameNStart = frameN  # exact frame index
            highlossframe2.setAutoDraw(True)
        
        # *lowlossframe1* updates
        if t >= 0.0 and lowlossframe1.status == NOT_STARTED:
            # keep track of start time/frame for later
            lowlossframe1.tStart = t  # underestimates by a little under one frame
            lowlossframe1.frameNStart = frameN  # exact frame index
            lowlossframe1.setAutoDraw(True)
        
        # *lowlossframe2* updates
        if t >= 0.0 and lowlossframe2.status == NOT_STARTED:
            # keep track of start time/frame for later
            lowlossframe2.tStart = t  # underestimates by a little under one frame
            lowlossframe2.frameNStart = frameN  # exact frame index
            lowlossframe2.setAutoDraw(True)
        
        # *highlosstoplabel* updates
        if t >= 0.0 and highlosstoplabel.status == NOT_STARTED:
            # keep track of start time/frame for later
            highlosstoplabel.tStart = t  # underestimates by a little under one frame
            highlosstoplabel.frameNStart = frameN  # exact frame index
            highlosstoplabel.setAutoDraw(True)
        
        # *highlossbottomlabel* updates
        if t >= 0.0 and highlossbottomlabel.status == NOT_STARTED:
            # keep track of start time/frame for later
            highlossbottomlabel.tStart = t  # underestimates by a little under one frame
            highlossbottomlabel.frameNStart = frameN  # exact frame index
            highlossbottomlabel.setAutoDraw(True)
        
        # *lowlosstoplabel* updates
        if t >= 0.0 and lowlosstoplabel.status == NOT_STARTED:
            # keep track of start time/frame for later
            lowlosstoplabel.tStart = t  # underestimates by a little under one frame
            lowlosstoplabel.frameNStart = frameN  # exact frame index
            lowlosstoplabel.setAutoDraw(True)
        
        # *lowlossbottomlabel* updates
        if t >= 0.0 and lowlossbottomlabel.status == NOT_STARTED:
            # keep track of start time/frame for later
            lowlossbottomlabel.tStart = t  # underestimates by a little under one frame
            lowlossbottomlabel.frameNStart = frameN  # exact frame index
            lowlossbottomlabel.setAutoDraw(True)
        
        # *frameDirText_3* updates
        if t >= 0.0 and frameDirText_3.status == NOT_STARTED:
            # keep track of start time/frame for later
            frameDirText_3.tStart = t  # underestimates by a little under one frame
            frameDirText_3.frameNStart = frameN  # exact frame index
            frameDirText_3.setAutoDraw(True)
        
        # *highlossLabel* updates
        if t >= 0.0 and highlossLabel.status == NOT_STARTED:
            # keep track of start time/frame for later
            highlossLabel.tStart = t  # underestimates by a little under one frame
            highlossLabel.frameNStart = frameN  # exact frame index
            highlossLabel.setAutoDraw(True)
        
        # *lowlosslabel* updates
        if t >= 0.0 and lowlosslabel.status == NOT_STARTED:
            # keep track of start time/frame for later
            lowlosslabel.tStart = t  # underestimates by a little under one frame
            lowlosslabel.frameNStart = frameN  # exact frame index
            lowlosslabel.setAutoDraw(True)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in lossDirsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    #-------Ending Routine "lossDirs"-------
    for thisComponent in lossDirsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "lossDirs" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 1 repeats of 'pracBlock'

# get names of stimulus parameters
if pracBlock.trialList in ([], [None], None):  params = []
else:  params = pracBlock.trialList[0].keys()
# save data for this loop
pracBlock.saveAsExcel(filename + '.xlsx', sheetName='pracBlock',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])

# set up handler to look after randomisation of conditions etc
taskBlock = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='taskBlock')
thisExp.addLoop(taskBlock)  # add the loop to the experiment
thisTaskBlock = taskBlock.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb=thisTaskBlock.rgb)
if thisTaskBlock != None:
    for paramName in thisTaskBlock.keys():
        exec(paramName + '= thisTaskBlock.' + paramName)

for thisTaskBlock in taskBlock:
    currentLoop = taskBlock
    # abbreviate parameter names if possible (e.g. rgb = thisTaskBlock.rgb)
    if thisTaskBlock != None:
        for paramName in thisTaskBlock.keys():
            exec(paramName + '= thisTaskBlock.' + paramName)
    
    #------Prepare to start Routine "instructions"-------
    t = 0
    instructionsClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    
    advance = event.BuilderKeyResponse()  # create an object of type KeyResponse
    advance.status = NOT_STARTED
    
    
    # keep track of which components have finished
    instructionsComponents = []
    instructionsComponents.append(instr_text)
    instructionsComponents.append(advance)
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    #-------Start Routine "instructions"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = instructionsClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        
        # *instr_text* updates
        if t >= 0.0 and instr_text.status == NOT_STARTED:
            # keep track of start time/frame for later
            instr_text.tStart = t  # underestimates by a little under one frame
            instr_text.frameNStart = frameN  # exact frame index
            instr_text.setAutoDraw(True)
        
        # *advance* updates
        if t >= 0.0 and advance.status == NOT_STARTED:
            # keep track of start time/frame for later
            advance.tStart = t  # underestimates by a little under one frame
            advance.frameNStart = frameN  # exact frame index
            advance.status = STARTED
            # keyboard checking is just starting
            event.clearEvents(eventType='keyboard')
        if advance.status == STARTED:
            theseKeys = event.getKeys(keyList=['space'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                # a response ends the routine
                continueRoutine = False
        
        
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    #-------Ending Routine "instructions"-------
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    
    
    
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    #------Prepare to start Routine "waitForScanner"-------
    t = 0
    waitForScannerClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    if expInfo['mriMode'] != 'Off':
        if trigger == 'usb':
            vol = launchScan(win, MR_settings, 
                  globalClock=fmriClock, 
                  mode=expInfo['mriMode'])
        elif trigger == 'parallel':
            address = 0x378
            #parallel.setPortAddress(0x378)
            wait_msg = "Waiting for scanner..."
            pinStatus = winioport.inp(address)
            waitMsgStim = visual.TextStim(win, color='DarkGray', text=wait_msg)
            waitMsgStim.draw()
            win.flip()
            while True:
                if pinStatus != winioport.inp(address):
                   break
                   # start exp when pin values change
            fmriClock.reset()
            logging.exp('parallel trigger: start of scan')
            win.flip()  # blank the screen on first sync pulse received
    
    expInfo['triggerWallTime'] = time.ctime()
    
    # keep track of which components have finished
    waitForScannerComponents = []
    for thisComponent in waitForScannerComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    #-------Start Routine "waitForScanner"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = waitForScannerClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in waitForScannerComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    #-------Ending Routine "waitForScanner"-------
    for thisComponent in waitForScannerComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    routineTimer.reset()
    # the Routine "waitForScanner" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    runs = data.TrialHandler(nReps=1, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditions/Runs.xlsx'),
        seed=None, name='runs')
    thisExp.addLoop(runs)  # add the loop to the experiment
    thisRun = runs.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb=thisRun.rgb)
    if thisRun != None:
        for paramName in thisRun.keys():
            exec(paramName + '= thisRun.' + paramName)
    
    for thisRun in runs:
        currentLoop = runs
        # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
        if thisRun != None:
            for paramName in thisRun.keys():
                exec(paramName + '= thisRun.' + paramName)
        
        #------Prepare to start Routine "prepRun"-------
        t = 0
        prepRunClock.reset()  # clock 
        frameN = -1
        routineTimer.add(0.300000)
        # update component parameters for each repeat
        if Runs == 'Run1':
            trialOrder = milConditionsFile1
        if Runs == 'Run2':
            trialOrder = milConditionsFile2
        if Runs == 'Run3':
            trialOrder = milConditionsFile3
        if Runs == 'Run4':
            trialOrder = milConditionsFile4
        # keep track of which components have finished
        prepRunComponents = []
        prepRunComponents.append(startFix1)
        for thisComponent in prepRunComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        
        #-------Start Routine "prepRun"-------
        continueRoutine = True
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = prepRunClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            
            # *startFix1* updates
            if t >= 0.0 and startFix1.status == NOT_STARTED:
                # keep track of start time/frame for later
                startFix1.tStart = t  # underestimates by a little under one frame
                startFix1.frameNStart = frameN  # exact frame index
                startFix1.setAutoDraw(True)
            if startFix1.status == STARTED and t >= (0.0 + (0.3-win.monitorFramePeriod*0.75)): #most of one frame period left
                startFix1.setAutoDraw(False)
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prepRunComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                core.quit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        #-------Ending Routine "prepRun"-------
        for thisComponent in prepRunComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler(nReps=1, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions(trialOrder),
            seed=None, name='trials')
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb=thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial.keys():
                exec(paramName + '= thisTrial.' + paramName)
        
        for thisTrial in trials:
            currentLoop = trials
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial.keys():
                    exec(paramName + '= thisTrial.' + paramName)
            
            #------Prepare to start Routine "cue"-------
            t = 0
            cueClock.reset()  # clock 
            frameN = -1
            routineTimer.add(2.500000)
            # update component parameters for each repeat
            if condition == "lowgain":
                corrvalue=0.25
                incorrvalue=0
                topLabel="+$%.2f"%(abs(corrvalue))
                bottomLabel="+$%.2f"%(abs(incorrvalue))
                framecolor=lowgainColor
            
            if condition == "highgain":
                corrvalue=0.50
                incorrvalue=0
                topLabel="+$%.2f"%(abs(corrvalue))
                bottomLabel="+$%.2f"%(abs(incorrvalue))
                framecolor=highgainColor
            
            if condition == "lowloss":
                corrvalue=0
                incorrvalue=-0.25
                topLabel="-$%.2f"%(abs(corrvalue))
                bottomLabel="-$%.2f"%(abs(incorrvalue))
                framecolor=lowlossColor
            
            if condition == "highloss":
                corrvalue=0
                incorrvalue=-0.5
                topLabel="-$%.2f"%(abs(corrvalue))
                bottomLabel="-$%.2f"%(abs(incorrvalue))
                framecolor=highlossColor
            
            
            
            currentLoop.addData('frameColor', framecolor)
            frame1.setLineColor([0,0,0])
            optimalImg = cuePairs[condition]['optimalChoice']
            suboptimalImg = cuePairs[condition]['suboptimalChoice']
            currentLoop.addData('optimalImg', optimalImg)
            currentLoop.addData('suboptimalImg', suboptimalImg)
            
            # Assumes button box responses are '1' for index and '2' for middle finger.
            if np.random.uniform() > 0.5:
                cuePositions = {response_keys['left']:optimalImg, response_keys['right']:suboptimalImg}
            else:
                cuePositions =  {response_keys['left']:suboptimalImg, response_keys['right']:optimalImg}
            sidePositions = dict([(v, k) for (k, v) in cuePositions.iteritems()])
            optimalSide='left' if sidePositions[optimalImg]==response_keys['left'] else 'right'
            suboptimalSide='left' if sidePositions[suboptimalImg]==response_keys['left'] else 'right'
            currentLoop.addData('optimalSide', optimalSide)
            currentLoop.addData('suboptimalSide', suboptimalSide)
            currentLoop.addData('optimalResponse',sidePositions[optimalImg])
            currentLoop.addData('suboptimalResponse',sidePositions[suboptimalImg])
            leftImage = 'images/'+cuePositions[response_keys['left']]
            currentLoop.addData('leftImage', leftImage)
            rightImage = 'images/'+cuePositions[response_keys['right']]
            currentLoop.addData('rightImage', rightImage)
            
            # Set the "correct" response by choosing the one that will give the better outcome.
            # That's a higher probability of winning in gain conditions and a lower probability 
            # of losing in loss conditions. Used for analysis, not for choosing outcomes.
            correct = sidePositions[optimalImg] 
            
            cueTimeAdded = False
            #chooseTimeAdded = False
            responseTimeAdded = False
            feedbackTimeAdded = False
            leftCue.setTex(leftImage)
            rightCue.setTex(rightImage)
            cueResp = event.BuilderKeyResponse()  # create an object of type KeyResponse
            cueResp.status = NOT_STARTED
            trialsClock.reset()
            topFrameText.setText(topLabel
)
            bottomFrameText.setText(bottomLabel)
            # Initialize cue end time for cues and selection indicator.
            # If no response is made in this time, the trial ends.
            # Otherwise, cueEndTime is updated to response
            # time plus the selection duration.
            cueEndTime = 2.5
            selectionIndicator.status = NOT_STARTED
            # keep track of which components have finished
            cueComponents = []
            cueComponents.append(frame1)
            cueComponents.append(frame2)
            cueComponents.append(leftCue)
            cueComponents.append(rightCue)
            cueComponents.append(cueResp)
            cueComponents.append(topFrameText)
            cueComponents.append(bottomFrameText)
            for thisComponent in cueComponents:
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            
            #-------Start Routine "cue"-------
            continueRoutine = True
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = cueClock.getTime()
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                
                # *frame1* updates
                if t >= 0.0 and frame1.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    frame1.tStart = t  # underestimates by a little under one frame
                    frame1.frameNStart = frameN  # exact frame index
                    frame1.setAutoDraw(True)
                if frame1.status == STARTED and t >= (0.0 + (2.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                    frame1.setAutoDraw(False)
                
                # *frame2* updates
                if t >= 0.0 and frame2.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    frame2.tStart = t  # underestimates by a little under one frame
                    frame2.frameNStart = frameN  # exact frame index
                    frame2.setAutoDraw(True)
                if frame2.status == STARTED and t >= (0.0 + (2.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                    frame2.setAutoDraw(False)
                
                if not cueTimeAdded:
                    currentLoop.addData('cueTime', fmriClock.getTime())
                    cueTimeAdded = True
                #if not chooseTimeAdded and chooseText.status == STARTED:
                #    currentLoop.addData('chooseTime', fmriClock.getTime())
                #    chooseTimeAdded = True
                if not responseTimeAdded and len(cueResp.keys):
                    currentLoop.addData('responseTime', fmriClock.getTime())
                    responseTimeAdded = True
                
                
                # *leftCue* updates
                if t >= 0.0 and leftCue.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    leftCue.tStart = t  # underestimates by a little under one frame
                    leftCue.frameNStart = frameN  # exact frame index
                    leftCue.setAutoDraw(True)
                if leftCue.status == STARTED and t >= (0.0 + (2.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                    leftCue.setAutoDraw(False)
                
                # *rightCue* updates
                if t >= 0.0 and rightCue.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    rightCue.tStart = t  # underestimates by a little under one frame
                    rightCue.frameNStart = frameN  # exact frame index
                    rightCue.setAutoDraw(True)
                if rightCue.status == STARTED and t >= (0.0 + (2.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                    rightCue.setAutoDraw(False)
                
                # *cueResp* updates
                if t >= 0 and cueResp.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    cueResp.tStart = t  # underestimates by a little under one frame
                    cueResp.frameNStart = frameN  # exact frame index
                    cueResp.status = STARTED
                    # AllowedKeys looks like a variable named `allowed_keys`
                    if not 'allowed_keys' in locals():
                        logging.error('AllowedKeys variable `allowed_keys` is not defined.')
                        core.quit()
                    if not type(allowed_keys) in [list, tuple, np.ndarray]:
                        if not isinstance(allowed_keys, basestring):
                            logging.error('AllowedKeys variable `allowed_keys` is not string- or list-like.')
                            core.quit()
                        elif not ',' in allowed_keys: allowed_keys = (allowed_keys,)
                        else:  allowed_keys = eval(allowed_keys)
                    # keyboard checking is just starting
                    win.callOnFlip(cueResp.clock.reset)  # t=0 on next screen flip
                    event.clearEvents(eventType='keyboard')
                if cueResp.status == STARTED and t >= (0 + (2.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                    cueResp.status = STOPPED
                if cueResp.status == STARTED:
                    theseKeys = event.getKeys(keyList=list(allowed_keys))
                    
                    # check for quit:
                    if "escape" in theseKeys:
                        endExpNow = True
                    if len(theseKeys) > 0:  # at least one key was pressed
                        if cueResp.keys == []:  # then this was the first keypress
                            cueResp.keys = theseKeys[0]  # just the first key pressed
                            cueResp.rt = cueResp.clock.getTime()
                            # was this 'correct'?
                            if (cueResp.keys == str(correct)) or (cueResp.keys == correct):
                                cueResp.corr = 1
                            else:
                                cueResp.corr = 0
                
                
                # *topFrameText* updates
                if t >= 0.0 and topFrameText.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    topFrameText.tStart = t  # underestimates by a little under one frame
                    topFrameText.frameNStart = frameN  # exact frame index
                    topFrameText.setAutoDraw(True)
                if topFrameText.status == STARTED and t >= (0.0 + (2.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                    topFrameText.setAutoDraw(False)
                
                # *bottomFrameText* updates
                if t >= 0.0 and bottomFrameText.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    bottomFrameText.tStart = t  # underestimates by a little under one frame
                    bottomFrameText.frameNStart = frameN  # exact frame index
                    bottomFrameText.setAutoDraw(True)
                if bottomFrameText.status == STARTED and t >= (0.0 + (2.5-win.monitorFramePeriod*0.75)): #most of one frame period left
                    bottomFrameText.setAutoDraw(False)
                # If the selection has been drawn for the selection duration,
                # or the trial should end (cutting off the selection duration), end the routine
                if t > cueEndTime:
                    continueRoutine = False
                
                # Start Drawing Selection Indicator if Selection has been made
                if cueResp.keys and selectionIndicator.status == NOT_STARTED:
                    cueComponents.append(selectionIndicator)
                    selectionIndicator.status = STARTED
                    selectionIndicator.setPos(selectionPosition(cueResp.keys))
                    selectionIndicator.setAutoDraw(True)
                    #if getSnapshots: win.getMovieFrame()
                    # Set End Time, clip to end of trial length (4 sec max)
                    # Don't worry about end of selection display in fMRI
                    #cueEndTime = t + selectionDuration
                
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in cueComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # check for quit (the Esc key)
                if endExpNow or event.getKeys(keyList=["escape"]):
                    core.quit()
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            #-------Ending Routine "cue"-------
            for thisComponent in cueComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            
            
            
            # check responses
            if cueResp.keys in ['', [], None]:  # No response was made
               cueResp.keys=None
               # was no response the correct answer?!
               if str(correct).lower() == 'none': cueResp.corr = 1  # correct non-response
               else: cueResp.corr = 0  # failed to respond (incorrectly)
            # store data for trials (TrialHandler)
            trials.addData('cueResp.keys',cueResp.keys)
            trials.addData('cueResp.corr', cueResp.corr)
            if cueResp.keys != None:  # we had a response
                trials.addData('cueResp.rt', cueResp.rt)
            
            
            
            #------Prepare to start Routine "outcomeDelay"-------
            t = 0
            outcomeDelayClock.reset()  # clock 
            frameN = -1
            # update component parameters for each repeat
            frame1_fix.setLineColor([0,0,0])
            topFrameText_fix.setText(topLabel
)
            bottomFrameText_fix.setText(bottomLabel)
            # keep track of which components have finished
            outcomeDelayComponents = []
            outcomeDelayComponents.append(frame1_fix)
            outcomeDelayComponents.append(frame2_fix)
            outcomeDelayComponents.append(outcomeDelayFix)
            outcomeDelayComponents.append(topFrameText_fix)
            outcomeDelayComponents.append(bottomFrameText_fix)
            for thisComponent in outcomeDelayComponents:
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            
            #-------Start Routine "outcomeDelay"-------
            continueRoutine = True
            while continueRoutine:
                # get current time
                t = outcomeDelayClock.getTime()
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *frame1_fix* updates
                if t >= 0.0 and frame1_fix.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    frame1_fix.tStart = t  # underestimates by a little under one frame
                    frame1_fix.frameNStart = frameN  # exact frame index
                    frame1_fix.setAutoDraw(True)
                if frame1_fix.status == STARTED and t >= (0.0 + (outcomeDelay-win.monitorFramePeriod*0.75)): #most of one frame period left
                    frame1_fix.setAutoDraw(False)
                
                # *frame2_fix* updates
                if t >= 0.0 and frame2_fix.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    frame2_fix.tStart = t  # underestimates by a little under one frame
                    frame2_fix.frameNStart = frameN  # exact frame index
                    frame2_fix.setAutoDraw(True)
                if frame2_fix.status == STARTED and t >= (0.0 + (outcomeDelay-win.monitorFramePeriod*0.75)): #most of one frame period left
                    frame2_fix.setAutoDraw(False)
                
                # *outcomeDelayFix* updates
                if t >= 0.0 and outcomeDelayFix.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    outcomeDelayFix.tStart = t  # underestimates by a little under one frame
                    outcomeDelayFix.frameNStart = frameN  # exact frame index
                    outcomeDelayFix.setAutoDraw(True)
                if outcomeDelayFix.status == STARTED and t >= (0.0 + (outcomeDelay-win.monitorFramePeriod*0.75)): #most of one frame period left
                    outcomeDelayFix.setAutoDraw(False)
                
                # *topFrameText_fix* updates
                if t >= 0.0 and topFrameText_fix.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    topFrameText_fix.tStart = t  # underestimates by a little under one frame
                    topFrameText_fix.frameNStart = frameN  # exact frame index
                    topFrameText_fix.setAutoDraw(True)
                if topFrameText_fix.status == STARTED and t >= (0.0 + (outcomeDelay-win.monitorFramePeriod*0.75)): #most of one frame period left
                    topFrameText_fix.setAutoDraw(False)
                
                # *bottomFrameText_fix* updates
                if t >= 0.0 and bottomFrameText_fix.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    bottomFrameText_fix.tStart = t  # underestimates by a little under one frame
                    bottomFrameText_fix.frameNStart = frameN  # exact frame index
                    bottomFrameText_fix.setAutoDraw(True)
                if bottomFrameText_fix.status == STARTED and t >= (0.0 + (outcomeDelay-win.monitorFramePeriod*0.75)): #most of one frame period left
                    bottomFrameText_fix.setAutoDraw(False)
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in outcomeDelayComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # check for quit (the Esc key)
                if endExpNow or event.getKeys(keyList=["escape"]):
                    core.quit()
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            #-------Ending Routine "outcomeDelay"-------
            for thisComponent in outcomeDelayComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # the Routine "outcomeDelay" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            #------Prepare to start Routine "outcome"-------
            t = 0
            outcomeClock.reset()  # clock 
            frameN = -1
            routineTimer.add(1.000000)
            # update component parameters for each repeat
            if cueResp.keys==sidePositions[optimalImg]:
                Choice=1
            elif cueResp.keys==sidePositions[suboptimalImg]:
                Choice=2
            else: 
                Choice="NA"
            
            Accuracy=cueResp.corr
            
            currentLoop.addData('Choice', Choice)
            currentLoop.addData('Accuracy', Accuracy)
            if condition == 'highgain':
                trueOperator = "+"
                falseOperator = "+"
            elif condition == 'lowgain':
                trueOperator = "+"
                falseOperator= "+"
            elif condition == 'highloss':
                trueOperator = "-"
                falseOperator= "-"
            else: # condition == 'lowloss'
            #    operator = ""
                trueOperator = "-"
                falseOperator = "-"
            if cueResp.keys:
                outcome = choose_outcome_from_cue(cueResp)
                currentLoop.addData('outcome', outcome)
                if outcome:
                    # If the outcome was true, display the change. 
                    # Format prettily using the operator instead of just using the change value.
                    change = corrvalue
                    outcome_msg = "%s$%.2f"%(trueOperator, abs(change))
                    outcome_image = None
                    Feedback=1
            
                else: 
                    # If outcome was False, the trial doesn't change anything.
                    # Still display the operator for trial condition
                    change = incorrvalue
                    outcome_msg = "%s$%.2f" % (falseOperator,abs(change))
                    outcome_image = None
                    Feedback=0
            else: # give incorrect outcome for missed trials
                change = incorrvalue
                currentLoop.addData('outcome', None)
                outcome_msg = "%s$%.2f" % (falseOperator,abs(change))
                outcome_image = None
                Feedback="NA"
            
            
            
            # Add a positive or negative change
            bank += change
            
            
            currentLoop.addData('Feedback', Feedback)
            currentLoop.addData('trialChange', change)
            currentLoop.addData('runningBankTotal', bank)
            currentLoop.addData('feedbackTime', fmriClock.getTime())
            frame1_fb.setLineColor([0,0,0])
            outcome_text.setText(outcome_msg)
            topFrameText_FB.setText(topLabel)
            bottomFrameText_FB.setText(bottomLabel)
            # keep track of which components have finished
            outcomeComponents = []
            outcomeComponents.append(frame1_fb)
            outcomeComponents.append(frame2_fb)
            outcomeComponents.append(outcome_text)
            outcomeComponents.append(topFrameText_FB)
            outcomeComponents.append(bottomFrameText_FB)
            for thisComponent in outcomeComponents:
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            
            #-------Start Routine "outcome"-------
            continueRoutine = True
            while continueRoutine and routineTimer.getTime() > 0:
                # get current time
                t = outcomeClock.getTime()
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                
                
                # *frame1_fb* updates
                if t >= 0.0 and frame1_fb.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    frame1_fb.tStart = t  # underestimates by a little under one frame
                    frame1_fb.frameNStart = frameN  # exact frame index
                    frame1_fb.setAutoDraw(True)
                if frame1_fb.status == STARTED and t >= (0.0 + (1-win.monitorFramePeriod*0.75)): #most of one frame period left
                    frame1_fb.setAutoDraw(False)
                
                # *frame2_fb* updates
                if t >= 0.0 and frame2_fb.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    frame2_fb.tStart = t  # underestimates by a little under one frame
                    frame2_fb.frameNStart = frameN  # exact frame index
                    frame2_fb.setAutoDraw(True)
                if frame2_fb.status == STARTED and t >= (0.0 + (1-win.monitorFramePeriod*0.75)): #most of one frame period left
                    frame2_fb.setAutoDraw(False)
                
                # *outcome_text* updates
                if t >= 0.0 and outcome_text.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    outcome_text.tStart = t  # underestimates by a little under one frame
                    outcome_text.frameNStart = frameN  # exact frame index
                    outcome_text.setAutoDraw(True)
                if outcome_text.status == STARTED and t >= (0.0 + (1-win.monitorFramePeriod*0.75)): #most of one frame period left
                    outcome_text.setAutoDraw(False)
                
                # *topFrameText_FB* updates
                if t >= 0.0 and topFrameText_FB.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    topFrameText_FB.tStart = t  # underestimates by a little under one frame
                    topFrameText_FB.frameNStart = frameN  # exact frame index
                    topFrameText_FB.setAutoDraw(True)
                if topFrameText_FB.status == STARTED and t >= (0.0 + (1-win.monitorFramePeriod*0.75)): #most of one frame period left
                    topFrameText_FB.setAutoDraw(False)
                
                # *bottomFrameText_FB* updates
                if t >= 0.0 and bottomFrameText_FB.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    bottomFrameText_FB.tStart = t  # underestimates by a little under one frame
                    bottomFrameText_FB.frameNStart = frameN  # exact frame index
                    bottomFrameText_FB.setAutoDraw(True)
                if bottomFrameText_FB.status == STARTED and t >= (0.0 + (1-win.monitorFramePeriod*0.75)): #most of one frame period left
                    bottomFrameText_FB.setAutoDraw(False)
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in outcomeComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # check for quit (the Esc key)
                if endExpNow or event.getKeys(keyList=["escape"]):
                    core.quit()
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            #-------Ending Routine "outcome"-------
            for thisComponent in outcomeComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            
            expInfo['bank'] = bank
            
            #------Prepare to start Routine "fixation"-------
            t = 0
            fixationClock.reset()  # clock 
            frameN = -1
            # update component parameters for each repeat
            currentLoop.addData('fixationTime', fmriClock.getTime())
            ##fixationDuration pre-specified in conditions file
            
            #fixationDuration = np.clip(np.random.normal(3,1),2,4)
            #trials.addData('expectedFixationDuration', fixationDuration)
            #logging.exp("Expected Fixation Duration: %f" % fixationDuration)
            # keep track of which components have finished
            fixationComponents = []
            fixationComponents.append(fixation_crosshair)
            fixationComponents.append(fixFrame1)
            fixationComponents.append(fixFrame2)
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            
            #-------Start Routine "fixation"-------
            continueRoutine = True
            while continueRoutine:
                # get current time
                t = fixationClock.getTime()
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixation_crosshair* updates
                if t >= 0.0 and fixation_crosshair.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    fixation_crosshair.tStart = t  # underestimates by a little under one frame
                    fixation_crosshair.frameNStart = frameN  # exact frame index
                    fixation_crosshair.setAutoDraw(True)
                if fixation_crosshair.status == STARTED and t >= (0.0 + (fixationDuration-win.monitorFramePeriod*0.75)): #most of one frame period left
                    fixation_crosshair.setAutoDraw(False)
                #if trialsClock.getTime() > 10.0:
                #    continueRoutine = False
                
                
                # *fixFrame1* updates
                if t >= 0.0 and fixFrame1.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    fixFrame1.tStart = t  # underestimates by a little under one frame
                    fixFrame1.frameNStart = frameN  # exact frame index
                    fixFrame1.setAutoDraw(True)
                if fixFrame1.status == STARTED and t >= (0.0 + (fixationDuration-win.monitorFramePeriod*0.75)): #most of one frame period left
                    fixFrame1.setAutoDraw(False)
                
                # *fixFrame2* updates
                if t >= 0.0 and fixFrame2.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    fixFrame2.tStart = t  # underestimates by a little under one frame
                    fixFrame2.frameNStart = frameN  # exact frame index
                    fixFrame2.setAutoDraw(True)
                if fixFrame2.status == STARTED and t >= (0.0 + (fixationDuration-win.monitorFramePeriod*0.75)): #most of one frame period left
                    fixFrame2.setAutoDraw(False)
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixationComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # check for quit (the Esc key)
                if endExpNow or event.getKeys(keyList=["escape"]):
                    core.quit()
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            #-------Ending Routine "fixation"-------
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            logging.exp("Measured Fixation Duration: %f" % t)
            logging.exp("trialsClock: %f" % trialsClock.getTime()) 
            #win.saveMovieFrames('thumb.png')
            # the Routine "fixation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 1 repeats of 'trials'
        
        # get names of stimulus parameters
        if trials.trialList in ([], [None], None):  params = []
        else:  params = trials.trialList[0].keys()
        # save data for this loop
        trials.saveAsExcel(filename + '.xlsx', sheetName='trials',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        #------Prepare to start Routine "breakscreen"-------
        t = 0
        breakscreenClock.reset()  # clock 
        frameN = -1
        # update component parameters for each repeat
        advanceScreen = event.BuilderKeyResponse()  # create an object of type KeyResponse
        advanceScreen.status = NOT_STARTED
        # keep track of which components have finished
        breakscreenComponents = []
        breakscreenComponents.append(breakText)
        breakscreenComponents.append(advanceScreen)
        for thisComponent in breakscreenComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        
        #-------Start Routine "breakscreen"-------
        continueRoutine = True
        while continueRoutine:
            # get current time
            t = breakscreenClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *breakText* updates
            if t >= 0.0 and breakText.status == NOT_STARTED:
                # keep track of start time/frame for later
                breakText.tStart = t  # underestimates by a little under one frame
                breakText.frameNStart = frameN  # exact frame index
                breakText.setAutoDraw(True)
            
            # *advanceScreen* updates
            if t >= 0.0 and advanceScreen.status == NOT_STARTED:
                # keep track of start time/frame for later
                advanceScreen.tStart = t  # underestimates by a little under one frame
                advanceScreen.frameNStart = frameN  # exact frame index
                advanceScreen.status = STARTED
                # keyboard checking is just starting
                event.clearEvents(eventType='keyboard')
            if advanceScreen.status == STARTED:
                theseKeys = event.getKeys(keyList=['y', 'n', 'left', 'right', 'space'])
                
                # check for quit:
                if "escape" in theseKeys:
                    endExpNow = True
                if len(theseKeys) > 0:  # at least one key was pressed
                    # a response ends the routine
                    continueRoutine = False
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in breakscreenComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                core.quit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        #-------Ending Routine "breakscreen"-------
        for thisComponent in breakscreenComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # the Routine "breakscreen" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1 repeats of 'runs'
    
    # get names of stimulus parameters
    if runs.trialList in ([], [None], None):  params = []
    else:  params = runs.trialList[0].keys()
    # save data for this loop
    runs.saveAsExcel(filename + '.xlsx', sheetName='runs',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    thisExp.nextEntry()
    
# completed 1 repeats of 'taskBlock'

# get names of stimulus parameters
if taskBlock.trialList in ([], [None], None):  params = []
else:  params = taskBlock.trialList[0].keys()
# save data for this loop
taskBlock.saveAsExcel(filename + '.xlsx', sheetName='taskBlock',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])

# set up handler to look after randomisation of conditions etc
ratingsBlock = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='ratingsBlock')
thisExp.addLoop(ratingsBlock)  # add the loop to the experiment
thisRatingsBlock = ratingsBlock.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb=thisRatingsBlock.rgb)
if thisRatingsBlock != None:
    for paramName in thisRatingsBlock.keys():
        exec(paramName + '= thisRatingsBlock.' + paramName)

for thisRatingsBlock in ratingsBlock:
    currentLoop = ratingsBlock
    # abbreviate parameter names if possible (e.g. rgb = thisRatingsBlock.rgb)
    if thisRatingsBlock != None:
        for paramName in thisRatingsBlock.keys():
            exec(paramName + '= thisRatingsBlock.' + paramName)
    
    #------Prepare to start Routine "ratingDir"-------
    t = 0
    ratingDirClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    advanceRating = event.BuilderKeyResponse()  # create an object of type KeyResponse
    advanceRating.status = NOT_STARTED
    # keep track of which components have finished
    ratingDirComponents = []
    ratingDirComponents.append(RatingDirs)
    ratingDirComponents.append(advanceRating)
    for thisComponent in ratingDirComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    #-------Start Routine "ratingDir"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = ratingDirClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *RatingDirs* updates
        if t >= 0.0 and RatingDirs.status == NOT_STARTED:
            # keep track of start time/frame for later
            RatingDirs.tStart = t  # underestimates by a little under one frame
            RatingDirs.frameNStart = frameN  # exact frame index
            RatingDirs.setAutoDraw(True)
        
        # *advanceRating* updates
        if t >= 0.0 and advanceRating.status == NOT_STARTED:
            # keep track of start time/frame for later
            advanceRating.tStart = t  # underestimates by a little under one frame
            advanceRating.frameNStart = frameN  # exact frame index
            advanceRating.status = STARTED
            # keyboard checking is just starting
            event.clearEvents(eventType='keyboard')
        if advanceRating.status == STARTED:
            theseKeys = event.getKeys(keyList=['space'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                # a response ends the routine
                continueRoutine = False
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ratingDirComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    #-------Ending Routine "ratingDir"-------
    for thisComponent in ratingDirComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "ratingDir" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    ratings = data.TrialHandler(nReps=1, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditions/PicList_Ratings.csv'),
        seed=None, name='ratings')
    thisExp.addLoop(ratings)  # add the loop to the experiment
    thisRating = ratings.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb=thisRating.rgb)
    if thisRating != None:
        for paramName in thisRating.keys():
            exec(paramName + '= thisRating.' + paramName)
    
    for thisRating in ratings:
        currentLoop = ratings
        # abbreviate parameter names if possible (e.g. rgb = thisRating.rgb)
        if thisRating != None:
            for paramName in thisRating.keys():
                exec(paramName + '= thisRating.' + paramName)
        
        #------Prepare to start Routine "rating"-------
        t = 0
        ratingClock.reset()  # clock 
        frameN = -1
        # update component parameters for each repeat
        image.setImage(os.path.join('images',Image))
        pic_rating.reset()
        # keep track of which components have finished
        ratingComponents = []
        ratingComponents.append(image)
        ratingComponents.append(pic_rating)
        ratingComponents.append(Question)
        for thisComponent in ratingComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        
        #-------Start Routine "rating"-------
        continueRoutine = True
        while continueRoutine:
            # get current time
            t = ratingClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image* updates
            if t >= 0.0 and image.status == NOT_STARTED:
                # keep track of start time/frame for later
                image.tStart = t  # underestimates by a little under one frame
                image.frameNStart = frameN  # exact frame index
                image.setAutoDraw(True)
            # *pic_rating* updates
            if t >= 0.0 and pic_rating.status == NOT_STARTED:
                # keep track of start time/frame for later
                pic_rating.tStart = t  # underestimates by a little under one frame
                pic_rating.frameNStart = frameN  # exact frame index
                pic_rating.setAutoDraw(True)
            continueRoutine &= pic_rating.noResponse  # a response ends the trial
            
            # *Question* updates
            if t >= 0.0 and Question.status == NOT_STARTED:
                # keep track of start time/frame for later
                Question.tStart = t  # underestimates by a little under one frame
                Question.frameNStart = frameN  # exact frame index
                Question.setAutoDraw(True)
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ratingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                core.quit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        #-------Ending Routine "rating"-------
        for thisComponent in ratingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store data for ratings (TrialHandler)
        ratings.addData('pic_rating.response', pic_rating.getRating())
        ratings.addData('pic_rating.rt', pic_rating.getRT())
        # the Routine "rating" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        #------Prepare to start Routine "ratingValence"-------
        t = 0
        ratingValenceClock.reset()  # clock 
        frameN = -1
        # update component parameters for each repeat
        image_2.setImage(os.path.join('images',Image))
        pic_rating_2.reset()
        # keep track of which components have finished
        ratingValenceComponents = []
        ratingValenceComponents.append(image_2)
        ratingValenceComponents.append(pic_rating_2)
        ratingValenceComponents.append(Question_2)
        for thisComponent in ratingValenceComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        
        #-------Start Routine "ratingValence"-------
        continueRoutine = True
        while continueRoutine:
            # get current time
            t = ratingValenceClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_2* updates
            if t >= 0.0 and image_2.status == NOT_STARTED:
                # keep track of start time/frame for later
                image_2.tStart = t  # underestimates by a little under one frame
                image_2.frameNStart = frameN  # exact frame index
                image_2.setAutoDraw(True)
            # *pic_rating_2* updates
            if t >= 0.0 and pic_rating_2.status == NOT_STARTED:
                # keep track of start time/frame for later
                pic_rating_2.tStart = t  # underestimates by a little under one frame
                pic_rating_2.frameNStart = frameN  # exact frame index
                pic_rating_2.setAutoDraw(True)
            continueRoutine &= pic_rating_2.noResponse  # a response ends the trial
            
            # *Question_2* updates
            if t >= 0.0 and Question_2.status == NOT_STARTED:
                # keep track of start time/frame for later
                Question_2.tStart = t  # underestimates by a little under one frame
                Question_2.frameNStart = frameN  # exact frame index
                Question_2.setAutoDraw(True)
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ratingValenceComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                core.quit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        #-------Ending Routine "ratingValence"-------
        for thisComponent in ratingValenceComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store data for ratings (TrialHandler)
        ratings.addData('pic_rating_2.response', pic_rating_2.getRating())
        ratings.addData('pic_rating_2.rt', pic_rating_2.getRT())
        # the Routine "ratingValence" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        #------Prepare to start Routine "ratingArousal"-------
        t = 0
        ratingArousalClock.reset()  # clock 
        frameN = -1
        # update component parameters for each repeat
        image_3.setImage(os.path.join('images',Image))
        pic_rating_3.reset()
        # keep track of which components have finished
        ratingArousalComponents = []
        ratingArousalComponents.append(image_3)
        ratingArousalComponents.append(pic_rating_3)
        ratingArousalComponents.append(Question_3)
        for thisComponent in ratingArousalComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        
        #-------Start Routine "ratingArousal"-------
        continueRoutine = True
        while continueRoutine:
            # get current time
            t = ratingArousalClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_3* updates
            if t >= 0.0 and image_3.status == NOT_STARTED:
                # keep track of start time/frame for later
                image_3.tStart = t  # underestimates by a little under one frame
                image_3.frameNStart = frameN  # exact frame index
                image_3.setAutoDraw(True)
            # *pic_rating_3* updates
            if t >= 0.0 and pic_rating_3.status == NOT_STARTED:
                # keep track of start time/frame for later
                pic_rating_3.tStart = t  # underestimates by a little under one frame
                pic_rating_3.frameNStart = frameN  # exact frame index
                pic_rating_3.setAutoDraw(True)
            continueRoutine &= pic_rating_3.noResponse  # a response ends the trial
            
            # *Question_3* updates
            if t >= 0.0 and Question_3.status == NOT_STARTED:
                # keep track of start time/frame for later
                Question_3.tStart = t  # underestimates by a little under one frame
                Question_3.frameNStart = frameN  # exact frame index
                Question_3.setAutoDraw(True)
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ratingArousalComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                core.quit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        #-------Ending Routine "ratingArousal"-------
        for thisComponent in ratingArousalComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store data for ratings (TrialHandler)
        ratings.addData('pic_rating_3.response', pic_rating_3.getRating())
        ratings.addData('pic_rating_3.rt', pic_rating_3.getRT())
        # the Routine "ratingArousal" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1 repeats of 'ratings'
    
    # get names of stimulus parameters
    if ratings.trialList in ([], [None], None):  params = []
    else:  params = ratings.trialList[0].keys()
    # save data for this loop
    ratings.saveAsExcel(filename + '.xlsx', sheetName='ratings',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    thisExp.nextEntry()
    
# completed 1 repeats of 'ratingsBlock'

# get names of stimulus parameters
if ratingsBlock.trialList in ([], [None], None):  params = []
else:  params = ratingsBlock.trialList[0].keys()
# save data for this loop
ratingsBlock.saveAsExcel(filename + '.xlsx', sheetName='ratingsBlock',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])

#------Prepare to start Routine "Done"-------
t = 0
DoneClock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
EndExperiment.setText('You finished the game!\n\nPlease tell the experimenter that you are finished!')
EndTask = event.BuilderKeyResponse()  # create an object of type KeyResponse
EndTask.status = NOT_STARTED
#end_msg = "You finished the game! You won $%s!"%(bank)
# keep track of which components have finished
DoneComponents = []
DoneComponents.append(EndExperiment)
DoneComponents.append(EndTask)
for thisComponent in DoneComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "Done"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = DoneClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *EndExperiment* updates
    if t >= 0.0 and EndExperiment.status == NOT_STARTED:
        # keep track of start time/frame for later
        EndExperiment.tStart = t  # underestimates by a little under one frame
        EndExperiment.frameNStart = frameN  # exact frame index
        EndExperiment.setAutoDraw(True)
    
    # *EndTask* updates
    if t >= 0.0 and EndTask.status == NOT_STARTED:
        # keep track of start time/frame for later
        EndTask.tStart = t  # underestimates by a little under one frame
        EndTask.frameNStart = frameN  # exact frame index
        EndTask.status = STARTED
        # keyboard checking is just starting
        event.clearEvents(eventType='keyboard')
    if EndTask.status == STARTED:
        theseKeys = event.getKeys(keyList=['y', 'n', 'left', 'right', 'space'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            # a response ends the routine
            continueRoutine = False
    
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in DoneComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

#-------Ending Routine "Done"-------
for thisComponent in DoneComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)

# the Routine "Done" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()




















# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort() # or data files will save again on exit
win.close()
core.quit()
