#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on Thu Aug 21 13:49:14 2025
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'RL_multises'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': ["1", "2", "3", "4", "5", "6", "7", "8", "OS-1", "OS-2", "OS-3", "OS-4", "OS-5", "OS-6", "OS-7", "OS-8", "pilot-1", "pilot-2", "pilot-3", "pilot-4", "pilot-5", "pilot-6", "pilot-7", "pilot-8"],
    'session': ["ses-1", "ses-2", "ses-3", "ses-4", "ses-5", "ses-6", "ses-7", "ses-8"],
    'startFromRun': ["1", "2"],
    'mode': ["pilot", "scan", "orientation"],
    'practice': ["yes", "no"],
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1728, 1117]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s_%s' % (expInfo['participant'], expName, expInfo['session'], expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/katharinaseitz/Documents/projects/year_1/RL-scancamp/reward_learning_task_multises_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='deg',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1,-1,-1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'deg'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('advanceScreenPress') is None:
        # initialise advanceScreenPress
        advanceScreenPress = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='advanceScreenPress',
        )
    if deviceManager.getDevice('pracPress') is None:
        # initialise pracPress
        pracPress = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='pracPress',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('pracResp') is None:
        # initialise pracResp
        pracResp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='pracResp',
        )
    if deviceManager.getDevice('check1Press') is None:
        # initialise check1Press
        check1Press = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='check1Press',
        )
    if deviceManager.getDevice('down1') is None:
        # initialise down1
        down1 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='down1',
        )
    if deviceManager.getDevice('phase2Press') is None:
        # initialise phase2Press
        phase2Press = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='phase2Press',
        )
    if deviceManager.getDevice('advanceScreenPress3') is None:
        # initialise advanceScreenPress3
        advanceScreenPress3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='advanceScreenPress3',
        )
    if deviceManager.getDevice('advanceScreen4') is None:
        # initialise advanceScreen4
        advanceScreen4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='advanceScreen4',
        )
    if deviceManager.getDevice('prac2Resp') is None:
        # initialise prac2Resp
        prac2Resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='prac2Resp',
        )
    if deviceManager.getDevice('check3Press') is None:
        # initialise check3Press
        check3Press = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='check3Press',
        )
    if deviceManager.getDevice('down2') is None:
        # initialise down2
        down2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='down2',
        )
    if deviceManager.getDevice('check2Press') is None:
        # initialise check2Press
        check2Press = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='check2Press',
        )
    if deviceManager.getDevice('down3') is None:
        # initialise down3
        down3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='down3',
        )
    if deviceManager.getDevice('check4Press') is None:
        # initialise check4Press
        check4Press = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='check4Press',
        )
    if deviceManager.getDevice('down4') is None:
        # initialise down4
        down4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='down4',
        )
    if deviceManager.getDevice('check5Press') is None:
        # initialise check5Press
        check5Press = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='check5Press',
        )
    if deviceManager.getDevice('down5') is None:
        # initialise down5
        down5 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='down5',
        )
    if deviceManager.getDevice('advanceScreenPress4') is None:
        # initialise advanceScreenPress4
        advanceScreenPress4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='advanceScreenPress4',
        )
    if deviceManager.getDevice('advanceToTrigger') is None:
        # initialise advanceToTrigger
        advanceToTrigger = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='advanceToTrigger',
        )
    if deviceManager.getDevice('scannerTriggerKey') is None:
        # initialise scannerTriggerKey
        scannerTriggerKey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='scannerTriggerKey',
        )
    if deviceManager.getDevice('cueResp') is None:
        # initialise cueResp
        cueResp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='cueResp',
        )
    if deviceManager.getDevice('advanceScreenPress_2') is None:
        # initialise advanceScreenPress_2
        advanceScreenPress_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='advanceScreenPress_2',
        )
    if deviceManager.getDevice('check6Press') is None:
        # initialise check6Press
        check6Press = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='check6Press',
        )
    if deviceManager.getDevice('check7Press') is None:
        # initialise check7Press
        check7Press = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='check7Press',
        )
    if deviceManager.getDevice('check8Press') is None:
        # initialise check8Press
        check8Press = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='check8Press',
        )
    if deviceManager.getDevice('check9Press') is None:
        # initialise check9Press
        check9Press = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='check9Press',
        )
    if deviceManager.getDevice('endTaskPress') is None:
        # initialise endTaskPress
        endTaskPress = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='endTaskPress',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "introDir" ---
    pracDirText = visual.TextStim(win=win, name='pracDirText',
        text='In this game, your job is to find the correct picture. Sometimes, you might have to guess. During the game, the correct picture might change. Try to choose the picture that is correct most of the time.\n\n\nPress the space bar to proceed.',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    advanceScreenPress = keyboard.Keyboard(deviceName='advanceScreenPress')
    # Run 'Begin Experiment' code from initializeVarCode
    import pandas as pd
    
    #are we doing practice
    if(expInfo["practice"] == "yes"):
        pracBlockRepeats = 1
    else:
        pracBlockRepeats = 0
        
    #pilot mode (on computer) vs scan mode in scanner
    if(expInfo["mode"] == "pilot" or expInfo["mode"] == "orientation"):
        leftKey = "left"
        rightKey = "right"
        responseKeys = dict(left = 'left', right = 'right')
        allowedKeys = ['left','right']
        PracCorrect = ["left", "left", "right", "right", "left", "right", 
                        "left", "left", "left", "right", "right", "left", 
                        "right", "left", "left"]
    else:
        leftKey = "2"
        rightKey = "3"
        responseKeys = dict(left = '2', right = '3')
        allowedKeys = ['2','3']
        PracCorrect = ["2", "2", "3", "3", "2", "3", "2", "2", "2", "3", "3", "2", "3", "2", "2"]
    
    
    
    #what is the participant's version
    if(not("pilot" in expInfo["participant"]) and not("OS" in expInfo["participant"])):
        version_df = pd.read_csv("stimuli_sets/which_vers.csv")
        version = version_df.loc[version_df['Participant'] == int(expInfo['participant']), expInfo['session']].values[0]
    else:
        if("pilot" in expInfo["participant"]):
            id_vers = int(expInfo["participant"][6:]) % 4
        else:
            id_vers = int(expInfo["participant"][3:]) % 4
        print(id_vers)
        if id_vers == 0:
            version = "A"
        elif id_vers == 1:
            version = "B"
        elif id_vers == 2:
            version = "C"
        else:
            version = "D"
    
    # --- Initialize components for Routine "pracDir1" ---
    choiceDirText = visual.TextStim(win=win, name='choiceDirText',
        text='There will be two pictures on the screen, one on the left and one on the right. Press the button with your pointer finger to choose the picture on the left, and press the button with your middle finger to choose the picture on the right. The pictures will change sides, but this does not affect whether or not the picture is correct.\n\n\n\n\n\n\n\n\n\n\nMake your choice as fast as you can. Once you choose, a box will show up on the screen. If you choose too late, your choice will not count.\n\nPractice selecting an image now by pressing with either finger. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac2Image = visual.ImageStim(
        win=win,
        name='prac2Image', 
        image='stimuli_sets/prac2.png', mask=None, anchor='center',
        ori=0.0, pos=(-4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    prac1Image = visual.ImageStim(
        win=win,
        name='prac1Image', 
        image='stimuli_sets/prac1.png', mask=None, anchor='center',
        ori=0.0, pos=(4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    pracPress = keyboard.Keyboard(deviceName='pracPress')
    
    # --- Initialize components for Routine "pracDir2" ---
    fixDirText = visual.TextStim(win=win, name='fixDirText',
        text='After you make a choice, you will see a + on the screen for a few seconds. Next, you will get feedback telling you if you are correct or incorrect.\n\n\n\n\n\nDo you have any questions?\n\nPress space bar to start the practice.',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "pracCue" ---
    pracLeftImg = visual.ImageStim(
        win=win,
        name='pracLeftImg', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    pracRightImg = visual.ImageStim(
        win=win,
        name='pracRightImg', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    pracResp = keyboard.Keyboard(deviceName='pracResp')
    # Run 'Begin Experiment' code from setRepeatsCode
    if expInfo["practice"] == "yes":
        pracOn = 1
    else:
        pracOn = 0
    # Run 'Begin Experiment' code from selectionCode
    selectionDuration = 2.5
    
    def selectionPosition(response):
        '''Set position of selectionCue'''
        side = .75 if response == responseKeys['right'] else -.75  # For fMRI, button 1 == left and button 2 == right, so positive will shift right and negative will shift left
        return [side * 180,0]
    
    # Cue Choice Indicator
    selectionIndicatorPrac = visual.Polygon(win, edges =4, ori=45, radius=180, 
                                    name = 'selectionIndicatorPrac', 
                                    lineColor = 'white', fillColor=None,
                                    units ='pix', lineWidth=3, interpolate=True)
    
    Choice=None
    
    # --- Initialize components for Routine "pracJitter" ---
    pracFix = visual.TextStim(win=win, name='pracFix',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from outcomeCode
    pracOutcome = None
    
    
    # --- Initialize components for Routine "pracOut" ---
    pracOutText = visual.TextStim(win=win, name='pracOutText',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    pracOutFix = visual.TextStim(win=win, name='pracOutFix',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "check1" ---
    check1Text = visual.TextStim(win=win, name='check1Text',
        text="Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. If you don't know, press the down-arrow key.",
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    check1ImageLeft = visual.ImageStim(
        win=win,
        name='check1ImageLeft', 
        image='stimuli_sets/prac2.png', mask=None, anchor='center',
        ori=0.0, pos=(-4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    check1ImageRight = visual.ImageStim(
        win=win,
        name='check1ImageRight', 
        image='stimuli_sets/prac1.png', mask=None, anchor='center',
        ori=0.0, pos=(4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    check1Press = keyboard.Keyboard(deviceName='check1Press')
    down1 = keyboard.Keyboard(deviceName='down1')
    
    # --- Initialize components for Routine "phase2Dir" ---
    phase2Instruc = visual.TextStim(win=win, name='phase2Instruc',
        text='Now, when you play the game, you can win money. \n\nYou will see 4 different choice pairs. Each pair can win or lose a different amount. \n\nPress the space bar to proceed.',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    phase2Press = keyboard.Keyboard(deviceName='phase2Press')
    
    # --- Initialize components for Routine "gainDirs" ---
    advanceScreenPress3 = keyboard.Keyboard(deviceName='advanceScreenPress3')
    gainDirText = visual.TextStim(win=win, name='gainDirText',
        text='Sometimes you can win money during the game!\n\nFor the HIGH WIN pair, you can win $1.00 if you are correct or win $0.00 if you are incorrect. \n\nFor the LOW WIN pair, you can win $0.20 if you are correct or win $0.00 if you are incorrect. \n\nThere will be a box around the pictures. The numbers in the box will tell you whether you can win a high or low amount. \n\nThe money you win will be paid to you as bonus at the end of the game, so try your best!',
        font='Arial',
        pos=(-6, 0), draggable=False, height=1.0, wrapWidth=20.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    highgainframe = visual.Rect(
        win=win, name='highgainframe',
        width=(8, 6)[0], height=(8, 6)[1],
        ori=0.0, pos=(10, 8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=-2.0, interpolate=True)
    lowgainframe = visual.Rect(
        win=win, name='lowgainframe',
        width=(8, 6)[0], height=(8, 6)[1],
        ori=0.0, pos=(10, -8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=-3.0, interpolate=True)
    highgainTopLabel = visual.TextStim(win=win, name='highgainTopLabel',
        text='+$1.00',
        font='Arial',
        pos=(10, 10.5), draggable=False, height=0.7, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    lowgainTopLabel = visual.TextStim(win=win, name='lowgainTopLabel',
        text='+$0.20',
        font='Arial',
        pos=(10, -5.5), draggable=False, height=0.7, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    highgainBottomLabel = visual.TextStim(win=win, name='highgainBottomLabel',
        text='+$0.00',
        font='Arial',
        pos=(10, 5.5), draggable=False, height=0.7, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    lowgainBottomLabel = visual.TextStim(win=win, name='lowgainBottomLabel',
        text='+$0.00',
        font='Arial',
        pos=(10, -10.5), draggable=False, height=0.7, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    highGainLabel = visual.TextStim(win=win, name='highGainLabel',
        text='HIGH GAIN',
        font='Arial',
        pos=(10, 8), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    lowGainLabel = visual.TextStim(win=win, name='lowGainLabel',
        text='LOW GAIN',
        font='Arial',
        pos=(10, -8), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    
    # --- Initialize components for Routine "lossDirs" ---
    advanceScreen4 = keyboard.Keyboard(deviceName='advanceScreen4')
    lossDirText = visual.TextStim(win=win, name='lossDirText',
        text='Sometimes you can lose money during the game!\n\nFor the HIGH LOSE pair, you can lose $0.00 if you are correct or lose $1.00 if you are incorrect.\n\nFor the LOW LOSE pair, you can lose $0.00 if you are correct or lose $0.20 if you are incorrect.\n\nThere will be a box around the pictures. The numbers in the box will tell you whether you can lose a high or low amount.\n\nThe money you lose will be taken away from the bonus money you win at the end of the game, so try your best!',
        font='Arial',
        pos=(-6, 0), draggable=False, height=1.0, wrapWidth=20.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    highLossFrame = visual.Rect(
        win=win, name='highLossFrame',
        width=(8, 6)[0], height=(8, 6)[1],
        ori=0.0, pos=(10, 8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=-2.0, interpolate=True)
    lowLossFrame = visual.Rect(
        win=win, name='lowLossFrame',
        width=(8, 6)[0], height=(8, 6)[1],
        ori=0.0, pos=(10, -8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=-3.0, interpolate=True)
    highLossTopLabel = visual.TextStim(win=win, name='highLossTopLabel',
        text='-$0.00',
        font='Arial',
        pos=(10, 10.5), draggable=False, height=0.7, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    lowLossTopLabel = visual.TextStim(win=win, name='lowLossTopLabel',
        text='-$0.00',
        font='Arial',
        pos=(10, -5.5), draggable=False, height=0.7, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    highLossBottomLabel = visual.TextStim(win=win, name='highLossBottomLabel',
        text='-$1.00',
        font='Arial',
        pos=(10, 5.5), draggable=False, height=0.7, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    lowLossBottomLabel = visual.TextStim(win=win, name='lowLossBottomLabel',
        text='-$0.20',
        font='Arial',
        pos=(10, -10.5), draggable=False, height=0.7, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    highLossLabel = visual.TextStim(win=win, name='highLossLabel',
        text='HIGH LOSE',
        font='Arial',
        pos=(10, 8), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    lowLossLabel = visual.TextStim(win=win, name='lowLossLabel',
        text='LOW GAIN',
        font='Arial',
        pos=(10, -8), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    # Run 'Begin Experiment' code from framePracOn
    frame = visual.Rect(
        win=win, name='frame1',
        width=(20,13)[0], height=(20,13)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "prac2Cue" ---
    prac2LeftCue = visual.ImageStim(
        win=win,
        name='prac2LeftCue', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    prac2RightCue = visual.ImageStim(
        win=win,
        name='prac2RightCue', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    prac2Resp = keyboard.Keyboard(deviceName='prac2Resp')
    # Run 'Begin Experiment' code from setCuesPrac
    cuePairPrac = {
        'highgain': {'optimalChoice':  'prac1.png', 'suboptimalChoice': 'prac2.png'}, 
        'lowgain': {'optimalChoice': 'prac3.png', 'suboptimalChoice': 'prac4.png'}, 
        'highloss': {'optimalChoice': 'prac5.png', 'suboptimalChoice': 'prac6.png'},
        'lowloss': {'optimalChoice': 'prac7.png', 'suboptimalChoice': 'prac8.png'}
    }
    
    ISI = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='ISI')
    # Run 'Begin Experiment' code from pracFrameLabels
    topLabel=None
    bottomLabel=None
    framecolor=None
    
    topFrameText = visual.TextStim(win=win, name='topFrameText',
        text='',
        font='Arial',
        pos=(0, 5.5), draggable=False, height=0.7, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    bottomFrameText = visual.TextStim(win=win, name='bottomFrameText',
        text='',
        font='Arial',
        pos=(0, -5.5), draggable=False, height=0.7, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    
    # --- Initialize components for Routine "prac2Jitter" ---
    prac2Fix = visual.TextStim(win=win, name='prac2Fix',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "prac2Out" ---
    # Run 'Begin Experiment' code from pracRightChoice
    Accuracy = None
    # Run 'Begin Experiment' code from definePracOut
    bank = 0
    def choose_prac_outcome_from_cue(response):
        '''Choose an outcome to display (if True, then the trial action will occur, if False it won't) 
        by setting a high or low threshold based on the response and image chosen, and then
        see if a random "roll" exceeds the threshold.
    
         The roll must be greater than threshold to return True, so a high probability will have a low
         threshold, and vice versa.'''
           
        if sidePositions[optimalImg] == prac2Resp.keys:
            return True if (optcorrect==1) else False
        elif sidePositions[suboptimalImg] == prac2Resp.keys:
            return True if (optcorrect==0) else False
    
    Accuracy_OptChoice=None
    outcomePracText = visual.TextStim(win=win, name='outcomePracText',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.75, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "check3" ---
    check3Text = visual.TextStim(win=win, name='check3Text',
        text="Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. If you don't know, press the down-arrow key.",
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    check3ImageLeft = visual.ImageStim(
        win=win,
        name='check3ImageLeft', 
        image='stimuli_sets/prac3.png', mask=None, anchor='center',
        ori=0.0, pos=(-4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    check3ImageRight = visual.ImageStim(
        win=win,
        name='check3ImageRight', 
        image='stimuli_sets/prac4.png', mask=None, anchor='center',
        ori=0.0, pos=(4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    check3Press = keyboard.Keyboard(deviceName='check3Press')
    down2 = keyboard.Keyboard(deviceName='down2')
    
    # --- Initialize components for Routine "check2" ---
    check2Text = visual.TextStim(win=win, name='check2Text',
        text="Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. If you don't know, press the down-arrow key.",
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    check2ImageLeft = visual.ImageStim(
        win=win,
        name='check2ImageLeft', 
        image='stimuli_sets/prac2.png', mask=None, anchor='center',
        ori=0.0, pos=(-4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    check2ImageRight = visual.ImageStim(
        win=win,
        name='check2ImageRight', 
        image='stimuli_sets/prac1.png', mask=None, anchor='center',
        ori=0.0, pos=(4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    check2Press = keyboard.Keyboard(deviceName='check2Press')
    down3 = keyboard.Keyboard(deviceName='down3')
    
    # --- Initialize components for Routine "check4" ---
    check4Text = visual.TextStim(win=win, name='check4Text',
        text="Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. If you don't know, press the down-arrow key.",
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    check4ImageLeft = visual.ImageStim(
        win=win,
        name='check4ImageLeft', 
        image='stimuli_sets/prac5.png', mask=None, anchor='center',
        ori=0.0, pos=(-4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    check4ImageRight = visual.ImageStim(
        win=win,
        name='check4ImageRight', 
        image='stimuli_sets/prac6.png', mask=None, anchor='center',
        ori=0.0, pos=(4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    check4Press = keyboard.Keyboard(deviceName='check4Press')
    down4 = keyboard.Keyboard(deviceName='down4')
    
    # --- Initialize components for Routine "check5" ---
    check5Text = visual.TextStim(win=win, name='check5Text',
        text="Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. If you don't know, press the down-arrow key.",
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    check5ImageLeft = visual.ImageStim(
        win=win,
        name='check5ImageLeft', 
        image='stimuli_sets/prac7.png', mask=None, anchor='center',
        ori=0.0, pos=(-4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    check5ImageRight = visual.ImageStim(
        win=win,
        name='check5ImageRight', 
        image='stimuli_sets/prac8.png', mask=None, anchor='center',
        ori=0.0, pos=(4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    check5Press = keyboard.Keyboard(deviceName='check5Press')
    down5 = keyboard.Keyboard(deviceName='down5')
    
    # --- Initialize components for Routine "summaryDirections" ---
    summary = visual.TextStim(win=win, name='summary',
        text="Directions to remember:  \n\n1. Try to choose the picture that gives you the best chance of winning money and avoiding losing money. \n\n2. Press the left button with your POINTER finger to select the image on the left side of the screen. Press the right button with your MIDDLE finger to select the image on the right side of the screen.  \n\n3. The pictures will sometimes appear on opposite sides of the screen. This does not change whether they will win or lose.  \n\n4. Make your choice when you see the pictures. If you choose after that, your response won't be counted.  \n\n5. The money that you win in this task will be YOURS TO KEEP.",
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=40.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from conditionsFiles
    import pandas as pd
    #load img set
    if(not("pilot" in expInfo["participant"]) and not("OS" in expInfo["participant"])):
        sets = pd.read_csv("stimuli_sets/which_set.csv")
        imgSet = sets.loc[sets['Participant'] == int(expInfo["participant"]), expInfo["session"]].values[0]
        
    else:    
        if("pilot" in expInfo["participant"]):
            id_num = int(expInfo["participant"][6:])
        else:
            id_num = int(expInfo["participant"][3:])
            id_num = int(expInfo["participant"][3:])
        id_num = expInfo["participant"][6:]
        imgSet = "set" + str(id_num)
    #load conditions
    if expInfo["startFromRun"] == "1":
        milConditionsFile1 = 'conditions/MIL_stakes_cond' + version + '_block1.csv' 
        milConditionsFile2 = 'conditions/MIL_stakes_cond' + version + '_block2.csv'
    else:
        milConditionsFile1 = 'conditions/MIL_stakes_cond' + version + '_block2.csv' 
        milConditionsFile2 = ''
    logging.exp("Using conditions file 1: %s" % milConditionsFile1)
    logging.exp("Using conditions file 2: %s" % milConditionsFile2)
    advanceScreenPress4 = keyboard.Keyboard(deviceName='advanceScreenPress4')
    
    # --- Initialize components for Routine "getReady" ---
    getReadyText = visual.TextStim(win=win, name='getReadyText',
        text='Get Ready!',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    advanceToTrigger = keyboard.Keyboard(deviceName='advanceToTrigger')
    # Run 'Begin Experiment' code from setRunFiles
    trialOrder = None
    
    # --- Initialize components for Routine "waitForScanner" ---
    # Run 'Begin Experiment' code from waitScannerCode
    waitForScannerClock = core.Clock()
    fmriClock = core.Clock()
    trigger = 'usb'
    
    
    
    ''' 
    #trigger = 'parallel'
    if trigger == 'parallel':
        from psychopy.contrib import parallel as winioport
        #from psychopy import parallel
    
    #import is failing
    elif trigger == 'usb':
        from psychopy.hardware.emulator import launchScan
        #
        # settings for launchScan:
        MR_settings = { 
            'TR': 2, # duration (sec) per volume
            'volumes': 215, # number of whole-brain 3D volumes / frames
            'sync': 'equal', # character to use as the sync timing event; assumed to come at start of a volume
            'skip': 0, # number of volumes lacking a sync pulse at start of scan (for T1 stabilization)
            }
    '''
    scannerTriggerKey = keyboard.Keyboard(deviceName='scannerTriggerKey')
    # Run 'Begin Experiment' code from frameOnCode
    frame = visual.Rect(
        win=win, name='frame1',
        width=(20,13)[0], height=(20,13)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=4.0,
        colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=-3.0, interpolate=True)
    sumTemplate = visual.ImageStim(
        win=win,
        name='sumTemplate', 
        image='summary.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(35, 18),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    hg1 = visual.ImageStim(
        win=win,
        name='hg1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-9.5, 1), draggable=False, size=(2.5, 2.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    hg2 = visual.ImageStim(
        win=win,
        name='hg2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-5.5, 1), draggable=False, size=(2.5, 2.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    lg1 = visual.ImageStim(
        win=win,
        name='lg1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-9.5, -5.3), draggable=False, size=(2.5, 2.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    lg2 = visual.ImageStim(
        win=win,
        name='lg2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-5.5, -5.3), draggable=False, size=(2.5, 2.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    hl1 = visual.ImageStim(
        win=win,
        name='hl1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(9.5, 1), draggable=False, size=(2.5, 2.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-9.0)
    hl2 = visual.ImageStim(
        win=win,
        name='hl2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(5.5, 1), draggable=False, size=(2.5, 2.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-10.0)
    ll1 = visual.ImageStim(
        win=win,
        name='ll1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(9.5, -5.3), draggable=False, size=(2.5, 2.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-11.0)
    ll2 = visual.ImageStim(
        win=win,
        name='ll2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(5.5, -5.3), draggable=False, size=(2.5, 2.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-12.0)
    
    # --- Initialize components for Routine "cue" ---
    startFixStatic = visual.TextStim(win=win, name='startFixStatic',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    leftCue = visual.ImageStim(
        win=win,
        name='leftCue', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    rightCue = visual.ImageStim(
        win=win,
        name='rightCue', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    cueResp = keyboard.Keyboard(deviceName='cueResp')
    # Run 'Begin Experiment' code from setCues
    import json
    jsonfile = 'Subject_Data/' + expInfo['participant'] + '_' + expInfo['session'] +'.json'
    # Assign Cues to Condition or load previous set.
    if(os.path.exists(jsonfile)):
        with open(jsonfile,'r') as f:
            cuePairs = json.load(f)
    else:
        imageList = 'conditions/'+ 'PicList_'+ version + '.csv'
        print(imageList)
        with open(imageList, 'r') as f:
            allImages=[s.strip() for s in f.readlines()]
            cuePairs = {
                'highgain': {'optimalChoice': imgSet + '_' + allImages[0], 'suboptimalChoice': imgSet + '_' +  allImages[1]}, 
                'lowgain': {'optimalChoice': imgSet + '_' + allImages[2], 'suboptimalChoice': imgSet + '_' + allImages[3]}, 
                'highloss': {'optimalChoice': imgSet + '_' + allImages[4], 'suboptimalChoice': imgSet +'_' +  allImages[5]},
                'lowloss': {'optimalChoice': imgSet + '_' + allImages[6], 'suboptimalChoice': imgSet + '_' + allImages[7]}
        }
    
        logging.info('Cue Pairs: %s' % cuePairs)
        expInfo['cuePairs'] = cuePairs
        with open(jsonfile, 'w') as f:
            f.write(json.dumps(cuePairs, sort_keys=True, indent=4))
    # Run 'Begin Experiment' code from frameLabels
    topLabel=None
    bottomLabel=None
    framecolor=None
    
    topFrameText = visual.TextStim(win=win, name='topFrameText',
        text='',
        font='Arial',
        pos=(0, 5.5), draggable=False, height=0.7, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    bottomFrameText = visual.TextStim(win=win, name='bottomFrameText',
        text='',
        font='Arial',
        pos=(0, -5.5), draggable=False, height=0.7, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    staticISI = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='staticISI')
    # Run 'Begin Experiment' code from manageTrialsLoop
    import time
    expInfo['expStartTime'] = time.ctime()
    
    TRIAL_DURATION = 10
    trialsClock = core.Clock()
    # Run 'Begin Experiment' code from selectionIndicatorCode
    selectionDuration = 3
    
    def selectionPosition(response):
        '''Set position of selectionCue'''
        side = .75 if response == responseKeys['right'] else -.75  # For fMRI, button 1 == left and button 2 == right, so positive will shift right and negative will shift left
        return [side * 180,0]
    
    # Cue Choice Indicator
    selectionIndicator = visual.Polygon(win, edges =4, ori=45, radius=180, 
                                    name = 'selectionIndicator', 
                                    lineColor = 'white', fillColor=None,
                                    units ='pix', lineWidth=3, interpolate=True)
    
    Choice=None
    
    # --- Initialize components for Routine "outcomeDelayRoutine" ---
    outcomeDelayFix = visual.TextStim(win=win, name='outcomeDelayFix',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "outcomeRoutine" ---
    # Run 'Begin Experiment' code from rightChoice
    Accuracy = None
    # Run 'Begin Experiment' code from defineOutcomes
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
    outcomeText = visual.TextStim(win=win, name='outcomeText',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.75, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "fixation" ---
    fixationCrosshair = visual.TextStim(win=win, name='fixationCrosshair',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "breakScreen" ---
    breakText = visual.TextStim(win=win, name='breakText',
        text='You finished this round!',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    advanceScreenPress_2 = keyboard.Keyboard(deviceName='advanceScreenPress_2')
    
    # --- Initialize components for Routine "check6" ---
    check6Text = visual.TextStim(win=win, name='check6Text',
        text='Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    check6ImageLeft = visual.ImageStim(
        win=win,
        name='check6ImageLeft', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    check6ImageRight = visual.ImageStim(
        win=win,
        name='check6ImageRight', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    check6Press = keyboard.Keyboard(deviceName='check6Press')
    
    # --- Initialize components for Routine "check7" ---
    check7Text = visual.TextStim(win=win, name='check7Text',
        text='Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    check7ImageLeft = visual.ImageStim(
        win=win,
        name='check7ImageLeft', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    check7ImageRight = visual.ImageStim(
        win=win,
        name='check7ImageRight', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    check7Press = keyboard.Keyboard(deviceName='check7Press')
    
    # --- Initialize components for Routine "check8" ---
    check8Text = visual.TextStim(win=win, name='check8Text',
        text='Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    check8ImageLeft = visual.ImageStim(
        win=win,
        name='check8ImageLeft', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    check8ImageRight = visual.ImageStim(
        win=win,
        name='check8ImageRight', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    check8Press = keyboard.Keyboard(deviceName='check8Press')
    
    # --- Initialize components for Routine "check9" ---
    check9Text = visual.TextStim(win=win, name='check9Text',
        text='Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    check9ImageLeft = visual.ImageStim(
        win=win,
        name='check9ImageLeft', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    check9ImageRight = visual.ImageStim(
        win=win,
        name='check9ImageRight', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(4, 0), draggable=False, size=(7, 7),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    check9Press = keyboard.Keyboard(deviceName='check9Press')
    
    # --- Initialize components for Routine "done" ---
    endExperiment = visual.TextStim(win=win, name='endExperiment',
        text='You finished the game!',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    endTaskPress = keyboard.Keyboard(deviceName='endTaskPress')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "introDir" ---
    # create an object to store info about Routine introDir
    introDir = data.Routine(
        name='introDir',
        components=[pracDirText, advanceScreenPress],
    )
    introDir.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for advanceScreenPress
    advanceScreenPress.keys = []
    advanceScreenPress.rt = []
    _advanceScreenPress_allKeys = []
    # store start times for introDir
    introDir.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    introDir.tStart = globalClock.getTime(format='float')
    introDir.status = STARTED
    thisExp.addData('introDir.started', introDir.tStart)
    introDir.maxDuration = None
    # keep track of which components have finished
    introDirComponents = introDir.components
    for thisComponent in introDir.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "introDir" ---
    introDir.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *pracDirText* updates
        
        # if pracDirText is starting this frame...
        if pracDirText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            pracDirText.frameNStart = frameN  # exact frame index
            pracDirText.tStart = t  # local t and not account for scr refresh
            pracDirText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pracDirText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'pracDirText.started')
            # update status
            pracDirText.status = STARTED
            pracDirText.setAutoDraw(True)
        
        # if pracDirText is active this frame...
        if pracDirText.status == STARTED:
            # update params
            pass
        
        # *advanceScreenPress* updates
        waitOnFlip = False
        
        # if advanceScreenPress is starting this frame...
        if advanceScreenPress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreenPress.frameNStart = frameN  # exact frame index
            advanceScreenPress.tStart = t  # local t and not account for scr refresh
            advanceScreenPress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreenPress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreenPress.started')
            # update status
            advanceScreenPress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreenPress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreenPress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreenPress.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreenPress.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _advanceScreenPress_allKeys.extend(theseKeys)
            if len(_advanceScreenPress_allKeys):
                advanceScreenPress.keys = _advanceScreenPress_allKeys[-1].name  # just the last key pressed
                advanceScreenPress.rt = _advanceScreenPress_allKeys[-1].rt
                advanceScreenPress.duration = _advanceScreenPress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=introDir,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            introDir.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in introDir.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "introDir" ---
    for thisComponent in introDir.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for introDir
    introDir.tStop = globalClock.getTime(format='float')
    introDir.tStopRefresh = tThisFlipGlobal
    thisExp.addData('introDir.stopped', introDir.tStop)
    # check responses
    if advanceScreenPress.keys in ['', [], None]:  # No response was made
        advanceScreenPress.keys = None
    thisExp.addData('advanceScreenPress.keys',advanceScreenPress.keys)
    if advanceScreenPress.keys != None:  # we had a response
        thisExp.addData('advanceScreenPress.rt', advanceScreenPress.rt)
        thisExp.addData('advanceScreenPress.duration', advanceScreenPress.duration)
    thisExp.nextEntry()
    # the Routine "introDir" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    pracBlock = data.TrialHandler2(
        name='pracBlock',
        nReps=pracBlockRepeats, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(pracBlock)  # add the loop to the experiment
    thisPracBlock = pracBlock.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPracBlock.rgb)
    if thisPracBlock != None:
        for paramName in thisPracBlock:
            globals()[paramName] = thisPracBlock[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPracBlock in pracBlock:
        pracBlock.status = STARTED
        if hasattr(thisPracBlock, 'status'):
            thisPracBlock.status = STARTED
        currentLoop = pracBlock
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPracBlock.rgb)
        if thisPracBlock != None:
            for paramName in thisPracBlock:
                globals()[paramName] = thisPracBlock[paramName]
        
        # --- Prepare to start Routine "pracDir1" ---
        # create an object to store info about Routine pracDir1
        pracDir1 = data.Routine(
            name='pracDir1',
            components=[choiceDirText, prac2Image, prac1Image, pracPress],
        )
        pracDir1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from selectionPracticeCode
        # Initialize cue end time for cues and selection indicator.
        # If no response is made in this time, the trial ends.
        # Otherwise, cueEndTime is updated to response
        # time plus the selection duration.
        
        selectionIndicatorPrac.status = NOT_STARTED
        t_started = False
        # create starting attributes for pracPress
        pracPress.keys = []
        pracPress.rt = []
        _pracPress_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'allowedKeys' in globals():
            allowedKeys = globals()['allowedKeys']
        # store start times for pracDir1
        pracDir1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pracDir1.tStart = globalClock.getTime(format='float')
        pracDir1.status = STARTED
        thisExp.addData('pracDir1.started', pracDir1.tStart)
        pracDir1.maxDuration = None
        # keep track of which components have finished
        pracDir1Components = pracDir1.components
        for thisComponent in pracDir1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pracDir1" ---
        pracDir1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPracBlock, 'status') and thisPracBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *choiceDirText* updates
            
            # if choiceDirText is starting this frame...
            if choiceDirText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                choiceDirText.frameNStart = frameN  # exact frame index
                choiceDirText.tStart = t  # local t and not account for scr refresh
                choiceDirText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(choiceDirText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'choiceDirText.started')
                # update status
                choiceDirText.status = STARTED
                choiceDirText.setAutoDraw(True)
            
            # if choiceDirText is active this frame...
            if choiceDirText.status == STARTED:
                # update params
                pass
            
            # *prac2Image* updates
            
            # if prac2Image is starting this frame...
            if prac2Image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac2Image.frameNStart = frameN  # exact frame index
                prac2Image.tStart = t  # local t and not account for scr refresh
                prac2Image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac2Image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac2Image.started')
                # update status
                prac2Image.status = STARTED
                prac2Image.setAutoDraw(True)
            
            # if prac2Image is active this frame...
            if prac2Image.status == STARTED:
                # update params
                pass
            
            # *prac1Image* updates
            
            # if prac1Image is starting this frame...
            if prac1Image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac1Image.frameNStart = frameN  # exact frame index
                prac1Image.tStart = t  # local t and not account for scr refresh
                prac1Image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac1Image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac1Image.started')
                # update status
                prac1Image.status = STARTED
                prac1Image.setAutoDraw(True)
            
            # if prac1Image is active this frame...
            if prac1Image.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from selectionPracticeCode
            # If the selection has been drawn for the selection duration,
            # or the trial should end (cutting off the selection duration), end the routine
            if t_started:
                if t > t_started + 5:
                    continueRoutine = False
            
            # Start Drawing Selection Indicator if Selection has been made
            if pracPress.keys and selectionIndicatorPrac.status == NOT_STARTED:
                #cueComponents.append(selectionIndicatorPrac)
                selectionIndicatorPrac.status = STARTED
                selectionIndicatorPrac.setPos(selectionPosition(pracPress.keys))
                selectionIndicatorPrac.setAutoDraw(True)
                t_started = core.Clock().getTime()
            
            # *pracPress* updates
            waitOnFlip = False
            
            # if pracPress is starting this frame...
            if pracPress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pracPress.frameNStart = frameN  # exact frame index
                pracPress.tStart = t  # local t and not account for scr refresh
                pracPress.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pracPress, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pracPress.started')
                # update status
                pracPress.status = STARTED
                # allowed keys looks like a variable named `allowedKeys`
                if not type(allowedKeys) in [list, tuple, np.ndarray]:
                    if not isinstance(allowedKeys, str):
                        allowedKeys = str(allowedKeys)
                    elif not ',' in allowedKeys:
                        allowedKeys = (allowedKeys,)
                    else:
                        allowedKeys = eval(allowedKeys)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(pracPress.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(pracPress.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if pracPress.status == STARTED and not waitOnFlip:
                theseKeys = pracPress.getKeys(keyList=list(allowedKeys), ignoreKeys=["escape"], waitRelease=False)
                _pracPress_allKeys.extend(theseKeys)
                if len(_pracPress_allKeys):
                    pracPress.keys = _pracPress_allKeys[-1].name  # just the last key pressed
                    pracPress.rt = _pracPress_allKeys[-1].rt
                    pracPress.duration = _pracPress_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=pracDir1,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                pracDir1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pracDir1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pracDir1" ---
        for thisComponent in pracDir1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pracDir1
        pracDir1.tStop = globalClock.getTime(format='float')
        pracDir1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pracDir1.stopped', pracDir1.tStop)
        # Run 'End Routine' code from selectionPracticeCode
        selectionIndicatorPrac.setAutoDraw(False)
        
        # check responses
        if pracPress.keys in ['', [], None]:  # No response was made
            pracPress.keys = None
        pracBlock.addData('pracPress.keys',pracPress.keys)
        if pracPress.keys != None:  # we had a response
            pracBlock.addData('pracPress.rt', pracPress.rt)
            pracBlock.addData('pracPress.duration', pracPress.duration)
        # the Routine "pracDir1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "pracDir2" ---
        # create an object to store info about Routine pracDir2
        pracDir2 = data.Routine(
            name='pracDir2',
            components=[fixDirText, key_resp],
        )
        pracDir2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for pracDir2
        pracDir2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pracDir2.tStart = globalClock.getTime(format='float')
        pracDir2.status = STARTED
        thisExp.addData('pracDir2.started', pracDir2.tStart)
        pracDir2.maxDuration = None
        # keep track of which components have finished
        pracDir2Components = pracDir2.components
        for thisComponent in pracDir2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pracDir2" ---
        pracDir2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPracBlock, 'status') and thisPracBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixDirText* updates
            
            # if fixDirText is starting this frame...
            if fixDirText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixDirText.frameNStart = frameN  # exact frame index
                fixDirText.tStart = t  # local t and not account for scr refresh
                fixDirText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixDirText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixDirText.started')
                # update status
                fixDirText.status = STARTED
                fixDirText.setAutoDraw(True)
            
            # if fixDirText is active this frame...
            if fixDirText.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=pracDir2,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                pracDir2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pracDir2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pracDir2" ---
        for thisComponent in pracDir2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pracDir2
        pracDir2.tStop = globalClock.getTime(format='float')
        pracDir2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pracDir2.stopped', pracDir2.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        pracBlock.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            pracBlock.addData('key_resp.rt', key_resp.rt)
            pracBlock.addData('key_resp.duration', key_resp.duration)
        # the Routine "pracDir2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        practice = data.TrialHandler2(
            name='practice',
            nReps=pracOn, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('conditions/practiceTrials.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(practice)  # add the loop to the experiment
        thisPractice = practice.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
        if thisPractice != None:
            for paramName in thisPractice:
                globals()[paramName] = thisPractice[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPractice in practice:
            practice.status = STARTED
            if hasattr(thisPractice, 'status'):
                thisPractice.status = STARTED
            currentLoop = practice
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
            if thisPractice != None:
                for paramName in thisPractice:
                    globals()[paramName] = thisPractice[paramName]
            
            # --- Prepare to start Routine "pracCue" ---
            # create an object to store info about Routine pracCue
            pracCue = data.Routine(
                name='pracCue',
                components=[pracLeftImg, pracRightImg, pracResp],
            )
            pracCue.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            pracLeftImg.setImage(leftImgPath)
            pracRightImg.setImage(rightImgPath)
            # create starting attributes for pracResp
            pracResp.keys = []
            pracResp.rt = []
            _pracResp_allKeys = []
            # allowedKeys looks like a variable, so make sure it exists locally
            if 'allowedKeys' in globals():
                allowedKeys = globals()['allowedKeys']
            # Run 'Begin Routine' code from selectionCode
            # Initialize cue end time for cues and selection indicator.
            # If no response is made in this time, the trial ends.
            # Otherwise, cueEndTime is updated to response
            # time plus the selection duration.
            cueEndTime = 2.5
            selectionIndicatorPrac.status = NOT_STARTED
            # store start times for pracCue
            pracCue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            pracCue.tStart = globalClock.getTime(format='float')
            pracCue.status = STARTED
            thisExp.addData('pracCue.started', pracCue.tStart)
            pracCue.maxDuration = None
            # keep track of which components have finished
            pracCueComponents = pracCue.components
            for thisComponent in pracCue.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "pracCue" ---
            pracCue.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.5:
                # if trial has changed, end Routine now
                if hasattr(thisPractice, 'status') and thisPractice.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *pracLeftImg* updates
                
                # if pracLeftImg is starting this frame...
                if pracLeftImg.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pracLeftImg.frameNStart = frameN  # exact frame index
                    pracLeftImg.tStart = t  # local t and not account for scr refresh
                    pracLeftImg.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracLeftImg, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracLeftImg.started')
                    # update status
                    pracLeftImg.status = STARTED
                    pracLeftImg.setAutoDraw(True)
                
                # if pracLeftImg is active this frame...
                if pracLeftImg.status == STARTED:
                    # update params
                    pass
                
                # if pracLeftImg is stopping this frame...
                if pracLeftImg.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracLeftImg.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        pracLeftImg.tStop = t  # not accounting for scr refresh
                        pracLeftImg.tStopRefresh = tThisFlipGlobal  # on global time
                        pracLeftImg.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracLeftImg.stopped')
                        # update status
                        pracLeftImg.status = FINISHED
                        pracLeftImg.setAutoDraw(False)
                
                # *pracRightImg* updates
                
                # if pracRightImg is starting this frame...
                if pracRightImg.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pracRightImg.frameNStart = frameN  # exact frame index
                    pracRightImg.tStart = t  # local t and not account for scr refresh
                    pracRightImg.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracRightImg, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracRightImg.started')
                    # update status
                    pracRightImg.status = STARTED
                    pracRightImg.setAutoDraw(True)
                
                # if pracRightImg is active this frame...
                if pracRightImg.status == STARTED:
                    # update params
                    pass
                
                # if pracRightImg is stopping this frame...
                if pracRightImg.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracRightImg.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        pracRightImg.tStop = t  # not accounting for scr refresh
                        pracRightImg.tStopRefresh = tThisFlipGlobal  # on global time
                        pracRightImg.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracRightImg.stopped')
                        # update status
                        pracRightImg.status = FINISHED
                        pracRightImg.setAutoDraw(False)
                
                # *pracResp* updates
                waitOnFlip = False
                
                # if pracResp is starting this frame...
                if pracResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pracResp.frameNStart = frameN  # exact frame index
                    pracResp.tStart = t  # local t and not account for scr refresh
                    pracResp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracResp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracResp.started')
                    # update status
                    pracResp.status = STARTED
                    # allowed keys looks like a variable named `allowedKeys`
                    if not type(allowedKeys) in [list, tuple, np.ndarray]:
                        if not isinstance(allowedKeys, str):
                            allowedKeys = str(allowedKeys)
                        elif not ',' in allowedKeys:
                            allowedKeys = (allowedKeys,)
                        else:
                            allowedKeys = eval(allowedKeys)
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(pracResp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(pracResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if pracResp is stopping this frame...
                if pracResp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracResp.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        pracResp.tStop = t  # not accounting for scr refresh
                        pracResp.tStopRefresh = tThisFlipGlobal  # on global time
                        pracResp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracResp.stopped')
                        # update status
                        pracResp.status = FINISHED
                        pracResp.status = FINISHED
                if pracResp.status == STARTED and not waitOnFlip:
                    theseKeys = pracResp.getKeys(keyList=list(allowedKeys), ignoreKeys=["escape"], waitRelease=False)
                    _pracResp_allKeys.extend(theseKeys)
                    if len(_pracResp_allKeys):
                        pracResp.keys = _pracResp_allKeys[0].name  # just the first key pressed
                        pracResp.rt = _pracResp_allKeys[0].rt
                        pracResp.duration = _pracResp_allKeys[0].duration
                        # was this correct?
                        if (pracResp.keys == str(PracCorrect)) or (pracResp.keys == PracCorrect):
                            pracResp.corr = 1
                        else:
                            pracResp.corr = 0
                # Run 'Each Frame' code from selectionCode
                # If the selection has been drawn for the selection duration,
                # or the trial should end (cutting off the selection duration), end the routine
                if t > cueEndTime:
                    continueRoutine = False
                
                # Start Drawing Selection Indicator if Selection has been made
                if pracResp.keys and selectionIndicatorPrac.status == NOT_STARTED:
                    #cueComponents.append(selectionIndicatorPrac)
                    selectionIndicatorPrac.status = STARTED
                    selectionIndicatorPrac.setPos(selectionPosition(pracResp.keys))
                    selectionIndicatorPrac.setAutoDraw(True)
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=pracCue,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    pracCue.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in pracCue.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "pracCue" ---
            for thisComponent in pracCue.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for pracCue
            pracCue.tStop = globalClock.getTime(format='float')
            pracCue.tStopRefresh = tThisFlipGlobal
            thisExp.addData('pracCue.stopped', pracCue.tStop)
            # check responses
            if pracResp.keys in ['', [], None]:  # No response was made
                pracResp.keys = None
                # was no response the correct answer?!
                if str(PracCorrect).lower() == 'none':
                   pracResp.corr = 1;  # correct non-response
                else:
                   pracResp.corr = 0;  # failed to respond (incorrectly)
            # store data for practice (TrialHandler)
            practice.addData('pracResp.keys',pracResp.keys)
            practice.addData('pracResp.corr', pracResp.corr)
            if pracResp.keys != None:  # we had a response
                practice.addData('pracResp.rt', pracResp.rt)
                practice.addData('pracResp.duration', pracResp.duration)
            # Run 'End Routine' code from selectionCode
            selectionIndicatorPrac.setAutoDraw(False)
            print(pracResp.corr == 1)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if pracCue.maxDurationReached:
                routineTimer.addTime(-pracCue.maxDuration)
            elif pracCue.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.500000)
            
            # --- Prepare to start Routine "pracJitter" ---
            # create an object to store info about Routine pracJitter
            pracJitter = data.Routine(
                name='pracJitter',
                components=[pracFix],
            )
            pracJitter.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from outcomeCode
            if pracResp.keys:
                if pracResp.keys == PracCorrect[practice.thisN]:
                    pracOutcome = "Correct!"
                else:
                    pracOutcome = "Incorrect!"
            else:
                pracOutcome = "Missed"
            # store start times for pracJitter
            pracJitter.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            pracJitter.tStart = globalClock.getTime(format='float')
            pracJitter.status = STARTED
            thisExp.addData('pracJitter.started', pracJitter.tStart)
            pracJitter.maxDuration = None
            # keep track of which components have finished
            pracJitterComponents = pracJitter.components
            for thisComponent in pracJitter.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "pracJitter" ---
            pracJitter.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.5:
                # if trial has changed, end Routine now
                if hasattr(thisPractice, 'status') and thisPractice.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *pracFix* updates
                
                # if pracFix is starting this frame...
                if pracFix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pracFix.frameNStart = frameN  # exact frame index
                    pracFix.tStart = t  # local t and not account for scr refresh
                    pracFix.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracFix, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracFix.started')
                    # update status
                    pracFix.status = STARTED
                    pracFix.setAutoDraw(True)
                
                # if pracFix is active this frame...
                if pracFix.status == STARTED:
                    # update params
                    pass
                
                # if pracFix is stopping this frame...
                if pracFix.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracFix.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        pracFix.tStop = t  # not accounting for scr refresh
                        pracFix.tStopRefresh = tThisFlipGlobal  # on global time
                        pracFix.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracFix.stopped')
                        # update status
                        pracFix.status = FINISHED
                        pracFix.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=pracJitter,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    pracJitter.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in pracJitter.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "pracJitter" ---
            for thisComponent in pracJitter.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for pracJitter
            pracJitter.tStop = globalClock.getTime(format='float')
            pracJitter.tStopRefresh = tThisFlipGlobal
            thisExp.addData('pracJitter.stopped', pracJitter.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if pracJitter.maxDurationReached:
                routineTimer.addTime(-pracJitter.maxDuration)
            elif pracJitter.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.500000)
            
            # --- Prepare to start Routine "pracOut" ---
            # create an object to store info about Routine pracOut
            pracOut = data.Routine(
                name='pracOut',
                components=[pracOutText, pracOutFix],
            )
            pracOut.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            pracOutText.setText(pracOutcome)
            # store start times for pracOut
            pracOut.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            pracOut.tStart = globalClock.getTime(format='float')
            pracOut.status = STARTED
            thisExp.addData('pracOut.started', pracOut.tStart)
            pracOut.maxDuration = None
            # keep track of which components have finished
            pracOutComponents = pracOut.components
            for thisComponent in pracOut.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "pracOut" ---
            pracOut.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.5:
                # if trial has changed, end Routine now
                if hasattr(thisPractice, 'status') and thisPractice.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *pracOutText* updates
                
                # if pracOutText is starting this frame...
                if pracOutText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pracOutText.frameNStart = frameN  # exact frame index
                    pracOutText.tStart = t  # local t and not account for scr refresh
                    pracOutText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracOutText, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracOutText.started')
                    # update status
                    pracOutText.status = STARTED
                    pracOutText.setAutoDraw(True)
                
                # if pracOutText is active this frame...
                if pracOutText.status == STARTED:
                    # update params
                    pass
                
                # if pracOutText is stopping this frame...
                if pracOutText.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracOutText.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        pracOutText.tStop = t  # not accounting for scr refresh
                        pracOutText.tStopRefresh = tThisFlipGlobal  # on global time
                        pracOutText.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracOutText.stopped')
                        # update status
                        pracOutText.status = FINISHED
                        pracOutText.setAutoDraw(False)
                
                # *pracOutFix* updates
                
                # if pracOutFix is starting this frame...
                if pracOutFix.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                    # keep track of start time/frame for later
                    pracOutFix.frameNStart = frameN  # exact frame index
                    pracOutFix.tStart = t  # local t and not account for scr refresh
                    pracOutFix.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pracOutFix, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracOutFix.started')
                    # update status
                    pracOutFix.status = STARTED
                    pracOutFix.setAutoDraw(True)
                
                # if pracOutFix is active this frame...
                if pracOutFix.status == STARTED:
                    # update params
                    pass
                
                # if pracOutFix is stopping this frame...
                if pracOutFix.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pracOutFix.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        pracOutFix.tStop = t  # not accounting for scr refresh
                        pracOutFix.tStopRefresh = tThisFlipGlobal  # on global time
                        pracOutFix.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pracOutFix.stopped')
                        # update status
                        pracOutFix.status = FINISHED
                        pracOutFix.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=pracOut,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    pracOut.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in pracOut.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "pracOut" ---
            for thisComponent in pracOut.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for pracOut
            pracOut.tStop = globalClock.getTime(format='float')
            pracOut.tStopRefresh = tThisFlipGlobal
            thisExp.addData('pracOut.stopped', pracOut.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if pracOut.maxDurationReached:
                routineTimer.addTime(-pracOut.maxDuration)
            elif pracOut.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.500000)
            # mark thisPractice as finished
            if hasattr(thisPractice, 'status'):
                thisPractice.status = FINISHED
            # if awaiting a pause, pause now
            if practice.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                practice.status = STARTED
            thisExp.nextEntry()
            
        # completed pracOn repeats of 'practice'
        practice.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if practice.trialList in ([], [None], None):
            params = []
        else:
            params = practice.trialList[0].keys()
        # save data for this loop
        practice.saveAsText(filename + '_practice.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "check1" ---
        # create an object to store info about Routine check1
        check1 = data.Routine(
            name='check1',
            components=[check1Text, check1ImageLeft, check1ImageRight, check1Press, down1],
        )
        check1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from check1Select
        # Initialize cue end time for cues and selection indicator.
        # If no response is made in this time, the trial ends.
        # Otherwise, cueEndTime is updated to response
        # time plus the selection duration.
        
        selectionIndicatorPrac.status = NOT_STARTED
        t_started = False
        # create starting attributes for check1Press
        check1Press.keys = []
        check1Press.rt = []
        _check1Press_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'allowedKeys' in globals():
            allowedKeys = globals()['allowedKeys']
        # create starting attributes for down1
        down1.keys = []
        down1.rt = []
        _down1_allKeys = []
        # store start times for check1
        check1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        check1.tStart = globalClock.getTime(format='float')
        check1.status = STARTED
        thisExp.addData('check1.started', check1.tStart)
        check1.maxDuration = None
        # keep track of which components have finished
        check1Components = check1.components
        for thisComponent in check1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "check1" ---
        check1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPracBlock, 'status') and thisPracBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *check1Text* updates
            
            # if check1Text is starting this frame...
            if check1Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check1Text.frameNStart = frameN  # exact frame index
                check1Text.tStart = t  # local t and not account for scr refresh
                check1Text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check1Text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check1Text.started')
                # update status
                check1Text.status = STARTED
                check1Text.setAutoDraw(True)
            
            # if check1Text is active this frame...
            if check1Text.status == STARTED:
                # update params
                pass
            
            # *check1ImageLeft* updates
            
            # if check1ImageLeft is starting this frame...
            if check1ImageLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check1ImageLeft.frameNStart = frameN  # exact frame index
                check1ImageLeft.tStart = t  # local t and not account for scr refresh
                check1ImageLeft.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check1ImageLeft, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check1ImageLeft.started')
                # update status
                check1ImageLeft.status = STARTED
                check1ImageLeft.setAutoDraw(True)
            
            # if check1ImageLeft is active this frame...
            if check1ImageLeft.status == STARTED:
                # update params
                pass
            
            # *check1ImageRight* updates
            
            # if check1ImageRight is starting this frame...
            if check1ImageRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check1ImageRight.frameNStart = frameN  # exact frame index
                check1ImageRight.tStart = t  # local t and not account for scr refresh
                check1ImageRight.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check1ImageRight, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check1ImageRight.started')
                # update status
                check1ImageRight.status = STARTED
                check1ImageRight.setAutoDraw(True)
            
            # if check1ImageRight is active this frame...
            if check1ImageRight.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from check1Select
            # If the selection has been drawn for the selection duration,
            # or the trial should end (cutting off the selection duration), end the routine
            if t_started:
                if t > t_started + 5:
                    continueRoutine = False
            
            # Start Drawing Selection Indicator if Selection has been made
            if check1Press.keys and selectionIndicatorPrac.status == NOT_STARTED:
                #cueComponents.append(selectionIndicatorPrac)
                selectionIndicatorPrac.status = STARTED
                selectionIndicatorPrac.setPos(selectionPosition(check1Press.keys))
                selectionIndicatorPrac.setAutoDraw(True)
                t_started = core.Clock().getTime()
            
            # *check1Press* updates
            waitOnFlip = False
            
            # if check1Press is starting this frame...
            if check1Press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check1Press.frameNStart = frameN  # exact frame index
                check1Press.tStart = t  # local t and not account for scr refresh
                check1Press.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check1Press, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check1Press.started')
                # update status
                check1Press.status = STARTED
                # allowed keys looks like a variable named `allowedKeys`
                if not type(allowedKeys) in [list, tuple, np.ndarray]:
                    if not isinstance(allowedKeys, str):
                        allowedKeys = str(allowedKeys)
                    elif not ',' in allowedKeys:
                        allowedKeys = (allowedKeys,)
                    else:
                        allowedKeys = eval(allowedKeys)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(check1Press.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(check1Press.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if check1Press.status == STARTED and not waitOnFlip:
                theseKeys = check1Press.getKeys(keyList=list(allowedKeys), ignoreKeys=["escape"], waitRelease=False)
                _check1Press_allKeys.extend(theseKeys)
                if len(_check1Press_allKeys):
                    check1Press.keys = _check1Press_allKeys[-1].name  # just the last key pressed
                    check1Press.rt = _check1Press_allKeys[-1].rt
                    check1Press.duration = _check1Press_allKeys[-1].duration
            
            # *down1* updates
            waitOnFlip = False
            
            # if down1 is starting this frame...
            if down1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                down1.frameNStart = frameN  # exact frame index
                down1.tStart = t  # local t and not account for scr refresh
                down1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(down1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'down1.started')
                # update status
                down1.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(down1.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(down1.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if down1.status == STARTED and not waitOnFlip:
                theseKeys = down1.getKeys(keyList=['down'], ignoreKeys=["escape"], waitRelease=False)
                _down1_allKeys.extend(theseKeys)
                if len(_down1_allKeys):
                    down1.keys = _down1_allKeys[-1].name  # just the last key pressed
                    down1.rt = _down1_allKeys[-1].rt
                    down1.duration = _down1_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=check1,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                check1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in check1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "check1" ---
        for thisComponent in check1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for check1
        check1.tStop = globalClock.getTime(format='float')
        check1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('check1.stopped', check1.tStop)
        # Run 'End Routine' code from check1Select
        selectionIndicatorPrac.setAutoDraw(False)
        
        # check responses
        if check1Press.keys in ['', [], None]:  # No response was made
            check1Press.keys = None
        pracBlock.addData('check1Press.keys',check1Press.keys)
        if check1Press.keys != None:  # we had a response
            pracBlock.addData('check1Press.rt', check1Press.rt)
            pracBlock.addData('check1Press.duration', check1Press.duration)
        # check responses
        if down1.keys in ['', [], None]:  # No response was made
            down1.keys = None
        pracBlock.addData('down1.keys',down1.keys)
        if down1.keys != None:  # we had a response
            pracBlock.addData('down1.rt', down1.rt)
            pracBlock.addData('down1.duration', down1.duration)
        # the Routine "check1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "phase2Dir" ---
        # create an object to store info about Routine phase2Dir
        phase2Dir = data.Routine(
            name='phase2Dir',
            components=[phase2Instruc, phase2Press],
        )
        phase2Dir.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for phase2Press
        phase2Press.keys = []
        phase2Press.rt = []
        _phase2Press_allKeys = []
        # store start times for phase2Dir
        phase2Dir.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        phase2Dir.tStart = globalClock.getTime(format='float')
        phase2Dir.status = STARTED
        thisExp.addData('phase2Dir.started', phase2Dir.tStart)
        phase2Dir.maxDuration = None
        # keep track of which components have finished
        phase2DirComponents = phase2Dir.components
        for thisComponent in phase2Dir.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "phase2Dir" ---
        phase2Dir.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPracBlock, 'status') and thisPracBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *phase2Instruc* updates
            
            # if phase2Instruc is starting this frame...
            if phase2Instruc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                phase2Instruc.frameNStart = frameN  # exact frame index
                phase2Instruc.tStart = t  # local t and not account for scr refresh
                phase2Instruc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(phase2Instruc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'phase2Instruc.started')
                # update status
                phase2Instruc.status = STARTED
                phase2Instruc.setAutoDraw(True)
            
            # if phase2Instruc is active this frame...
            if phase2Instruc.status == STARTED:
                # update params
                pass
            
            # *phase2Press* updates
            waitOnFlip = False
            
            # if phase2Press is starting this frame...
            if phase2Press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                phase2Press.frameNStart = frameN  # exact frame index
                phase2Press.tStart = t  # local t and not account for scr refresh
                phase2Press.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(phase2Press, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'phase2Press.started')
                # update status
                phase2Press.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(phase2Press.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(phase2Press.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if phase2Press.status == STARTED and not waitOnFlip:
                theseKeys = phase2Press.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _phase2Press_allKeys.extend(theseKeys)
                if len(_phase2Press_allKeys):
                    phase2Press.keys = _phase2Press_allKeys[-1].name  # just the last key pressed
                    phase2Press.rt = _phase2Press_allKeys[-1].rt
                    phase2Press.duration = _phase2Press_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=phase2Dir,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                phase2Dir.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in phase2Dir.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "phase2Dir" ---
        for thisComponent in phase2Dir.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for phase2Dir
        phase2Dir.tStop = globalClock.getTime(format='float')
        phase2Dir.tStopRefresh = tThisFlipGlobal
        thisExp.addData('phase2Dir.stopped', phase2Dir.tStop)
        # check responses
        if phase2Press.keys in ['', [], None]:  # No response was made
            phase2Press.keys = None
        pracBlock.addData('phase2Press.keys',phase2Press.keys)
        if phase2Press.keys != None:  # we had a response
            pracBlock.addData('phase2Press.rt', phase2Press.rt)
            pracBlock.addData('phase2Press.duration', phase2Press.duration)
        # the Routine "phase2Dir" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "gainDirs" ---
        # create an object to store info about Routine gainDirs
        gainDirs = data.Routine(
            name='gainDirs',
            components=[advanceScreenPress3, gainDirText, highgainframe, lowgainframe, highgainTopLabel, lowgainTopLabel, highgainBottomLabel, lowgainBottomLabel, highGainLabel, lowGainLabel],
        )
        gainDirs.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for advanceScreenPress3
        advanceScreenPress3.keys = []
        advanceScreenPress3.rt = []
        _advanceScreenPress3_allKeys = []
        highgainframe.setLineColor('white')
        lowgainframe.setLineColor('white')
        # Run 'Begin Routine' code from gainDirText_2
        gainDirText.alignText = 'left'
        # store start times for gainDirs
        gainDirs.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        gainDirs.tStart = globalClock.getTime(format='float')
        gainDirs.status = STARTED
        thisExp.addData('gainDirs.started', gainDirs.tStart)
        gainDirs.maxDuration = None
        # keep track of which components have finished
        gainDirsComponents = gainDirs.components
        for thisComponent in gainDirs.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "gainDirs" ---
        gainDirs.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPracBlock, 'status') and thisPracBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *advanceScreenPress3* updates
            waitOnFlip = False
            
            # if advanceScreenPress3 is starting this frame...
            if advanceScreenPress3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                advanceScreenPress3.frameNStart = frameN  # exact frame index
                advanceScreenPress3.tStart = t  # local t and not account for scr refresh
                advanceScreenPress3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(advanceScreenPress3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'advanceScreenPress3.started')
                # update status
                advanceScreenPress3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(advanceScreenPress3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(advanceScreenPress3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if advanceScreenPress3.status == STARTED and not waitOnFlip:
                theseKeys = advanceScreenPress3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _advanceScreenPress3_allKeys.extend(theseKeys)
                if len(_advanceScreenPress3_allKeys):
                    advanceScreenPress3.keys = _advanceScreenPress3_allKeys[-1].name  # just the last key pressed
                    advanceScreenPress3.rt = _advanceScreenPress3_allKeys[-1].rt
                    advanceScreenPress3.duration = _advanceScreenPress3_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *gainDirText* updates
            
            # if gainDirText is starting this frame...
            if gainDirText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gainDirText.frameNStart = frameN  # exact frame index
                gainDirText.tStart = t  # local t and not account for scr refresh
                gainDirText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gainDirText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gainDirText.started')
                # update status
                gainDirText.status = STARTED
                gainDirText.setAutoDraw(True)
            
            # if gainDirText is active this frame...
            if gainDirText.status == STARTED:
                # update params
                pass
            
            # *highgainframe* updates
            
            # if highgainframe is starting this frame...
            if highgainframe.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                highgainframe.frameNStart = frameN  # exact frame index
                highgainframe.tStart = t  # local t and not account for scr refresh
                highgainframe.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(highgainframe, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'highgainframe.started')
                # update status
                highgainframe.status = STARTED
                highgainframe.setAutoDraw(True)
            
            # if highgainframe is active this frame...
            if highgainframe.status == STARTED:
                # update params
                pass
            
            # *lowgainframe* updates
            
            # if lowgainframe is starting this frame...
            if lowgainframe.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lowgainframe.frameNStart = frameN  # exact frame index
                lowgainframe.tStart = t  # local t and not account for scr refresh
                lowgainframe.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lowgainframe, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lowgainframe.started')
                # update status
                lowgainframe.status = STARTED
                lowgainframe.setAutoDraw(True)
            
            # if lowgainframe is active this frame...
            if lowgainframe.status == STARTED:
                # update params
                pass
            
            # *highgainTopLabel* updates
            
            # if highgainTopLabel is starting this frame...
            if highgainTopLabel.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                highgainTopLabel.frameNStart = frameN  # exact frame index
                highgainTopLabel.tStart = t  # local t and not account for scr refresh
                highgainTopLabel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(highgainTopLabel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'highgainTopLabel.started')
                # update status
                highgainTopLabel.status = STARTED
                highgainTopLabel.setAutoDraw(True)
            
            # if highgainTopLabel is active this frame...
            if highgainTopLabel.status == STARTED:
                # update params
                pass
            
            # *lowgainTopLabel* updates
            
            # if lowgainTopLabel is starting this frame...
            if lowgainTopLabel.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lowgainTopLabel.frameNStart = frameN  # exact frame index
                lowgainTopLabel.tStart = t  # local t and not account for scr refresh
                lowgainTopLabel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lowgainTopLabel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lowgainTopLabel.started')
                # update status
                lowgainTopLabel.status = STARTED
                lowgainTopLabel.setAutoDraw(True)
            
            # if lowgainTopLabel is active this frame...
            if lowgainTopLabel.status == STARTED:
                # update params
                pass
            
            # *highgainBottomLabel* updates
            
            # if highgainBottomLabel is starting this frame...
            if highgainBottomLabel.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                highgainBottomLabel.frameNStart = frameN  # exact frame index
                highgainBottomLabel.tStart = t  # local t and not account for scr refresh
                highgainBottomLabel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(highgainBottomLabel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'highgainBottomLabel.started')
                # update status
                highgainBottomLabel.status = STARTED
                highgainBottomLabel.setAutoDraw(True)
            
            # if highgainBottomLabel is active this frame...
            if highgainBottomLabel.status == STARTED:
                # update params
                pass
            
            # *lowgainBottomLabel* updates
            
            # if lowgainBottomLabel is starting this frame...
            if lowgainBottomLabel.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lowgainBottomLabel.frameNStart = frameN  # exact frame index
                lowgainBottomLabel.tStart = t  # local t and not account for scr refresh
                lowgainBottomLabel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lowgainBottomLabel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lowgainBottomLabel.started')
                # update status
                lowgainBottomLabel.status = STARTED
                lowgainBottomLabel.setAutoDraw(True)
            
            # if lowgainBottomLabel is active this frame...
            if lowgainBottomLabel.status == STARTED:
                # update params
                pass
            
            # *highGainLabel* updates
            
            # if highGainLabel is starting this frame...
            if highGainLabel.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                highGainLabel.frameNStart = frameN  # exact frame index
                highGainLabel.tStart = t  # local t and not account for scr refresh
                highGainLabel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(highGainLabel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'highGainLabel.started')
                # update status
                highGainLabel.status = STARTED
                highGainLabel.setAutoDraw(True)
            
            # if highGainLabel is active this frame...
            if highGainLabel.status == STARTED:
                # update params
                pass
            
            # *lowGainLabel* updates
            
            # if lowGainLabel is starting this frame...
            if lowGainLabel.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lowGainLabel.frameNStart = frameN  # exact frame index
                lowGainLabel.tStart = t  # local t and not account for scr refresh
                lowGainLabel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lowGainLabel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lowGainLabel.started')
                # update status
                lowGainLabel.status = STARTED
                lowGainLabel.setAutoDraw(True)
            
            # if lowGainLabel is active this frame...
            if lowGainLabel.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=gainDirs,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                gainDirs.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in gainDirs.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "gainDirs" ---
        for thisComponent in gainDirs.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for gainDirs
        gainDirs.tStop = globalClock.getTime(format='float')
        gainDirs.tStopRefresh = tThisFlipGlobal
        thisExp.addData('gainDirs.stopped', gainDirs.tStop)
        # check responses
        if advanceScreenPress3.keys in ['', [], None]:  # No response was made
            advanceScreenPress3.keys = None
        pracBlock.addData('advanceScreenPress3.keys',advanceScreenPress3.keys)
        if advanceScreenPress3.keys != None:  # we had a response
            pracBlock.addData('advanceScreenPress3.rt', advanceScreenPress3.rt)
            pracBlock.addData('advanceScreenPress3.duration', advanceScreenPress3.duration)
        # the Routine "gainDirs" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "lossDirs" ---
        # create an object to store info about Routine lossDirs
        lossDirs = data.Routine(
            name='lossDirs',
            components=[advanceScreen4, lossDirText, highLossFrame, lowLossFrame, highLossTopLabel, lowLossTopLabel, highLossBottomLabel, lowLossBottomLabel, highLossLabel, lowLossLabel],
        )
        lossDirs.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for advanceScreen4
        advanceScreen4.keys = []
        advanceScreen4.rt = []
        _advanceScreen4_allKeys = []
        highLossFrame.setLineColor('white')
        lowLossFrame.setLineColor('white')
        # Run 'Begin Routine' code from lossTextCode
        lossDirText.alignText = 'left'
        # store start times for lossDirs
        lossDirs.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        lossDirs.tStart = globalClock.getTime(format='float')
        lossDirs.status = STARTED
        thisExp.addData('lossDirs.started', lossDirs.tStart)
        lossDirs.maxDuration = None
        # keep track of which components have finished
        lossDirsComponents = lossDirs.components
        for thisComponent in lossDirs.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "lossDirs" ---
        lossDirs.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPracBlock, 'status') and thisPracBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *advanceScreen4* updates
            waitOnFlip = False
            
            # if advanceScreen4 is starting this frame...
            if advanceScreen4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                advanceScreen4.frameNStart = frameN  # exact frame index
                advanceScreen4.tStart = t  # local t and not account for scr refresh
                advanceScreen4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(advanceScreen4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'advanceScreen4.started')
                # update status
                advanceScreen4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(advanceScreen4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(advanceScreen4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if advanceScreen4.status == STARTED and not waitOnFlip:
                theseKeys = advanceScreen4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _advanceScreen4_allKeys.extend(theseKeys)
                if len(_advanceScreen4_allKeys):
                    advanceScreen4.keys = _advanceScreen4_allKeys[-1].name  # just the last key pressed
                    advanceScreen4.rt = _advanceScreen4_allKeys[-1].rt
                    advanceScreen4.duration = _advanceScreen4_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *lossDirText* updates
            
            # if lossDirText is starting this frame...
            if lossDirText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lossDirText.frameNStart = frameN  # exact frame index
                lossDirText.tStart = t  # local t and not account for scr refresh
                lossDirText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lossDirText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lossDirText.started')
                # update status
                lossDirText.status = STARTED
                lossDirText.setAutoDraw(True)
            
            # if lossDirText is active this frame...
            if lossDirText.status == STARTED:
                # update params
                pass
            
            # *highLossFrame* updates
            
            # if highLossFrame is starting this frame...
            if highLossFrame.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                highLossFrame.frameNStart = frameN  # exact frame index
                highLossFrame.tStart = t  # local t and not account for scr refresh
                highLossFrame.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(highLossFrame, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'highLossFrame.started')
                # update status
                highLossFrame.status = STARTED
                highLossFrame.setAutoDraw(True)
            
            # if highLossFrame is active this frame...
            if highLossFrame.status == STARTED:
                # update params
                pass
            
            # *lowLossFrame* updates
            
            # if lowLossFrame is starting this frame...
            if lowLossFrame.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lowLossFrame.frameNStart = frameN  # exact frame index
                lowLossFrame.tStart = t  # local t and not account for scr refresh
                lowLossFrame.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lowLossFrame, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lowLossFrame.started')
                # update status
                lowLossFrame.status = STARTED
                lowLossFrame.setAutoDraw(True)
            
            # if lowLossFrame is active this frame...
            if lowLossFrame.status == STARTED:
                # update params
                pass
            
            # *highLossTopLabel* updates
            
            # if highLossTopLabel is starting this frame...
            if highLossTopLabel.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                highLossTopLabel.frameNStart = frameN  # exact frame index
                highLossTopLabel.tStart = t  # local t and not account for scr refresh
                highLossTopLabel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(highLossTopLabel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'highLossTopLabel.started')
                # update status
                highLossTopLabel.status = STARTED
                highLossTopLabel.setAutoDraw(True)
            
            # if highLossTopLabel is active this frame...
            if highLossTopLabel.status == STARTED:
                # update params
                pass
            
            # *lowLossTopLabel* updates
            
            # if lowLossTopLabel is starting this frame...
            if lowLossTopLabel.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lowLossTopLabel.frameNStart = frameN  # exact frame index
                lowLossTopLabel.tStart = t  # local t and not account for scr refresh
                lowLossTopLabel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lowLossTopLabel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lowLossTopLabel.started')
                # update status
                lowLossTopLabel.status = STARTED
                lowLossTopLabel.setAutoDraw(True)
            
            # if lowLossTopLabel is active this frame...
            if lowLossTopLabel.status == STARTED:
                # update params
                pass
            
            # *highLossBottomLabel* updates
            
            # if highLossBottomLabel is starting this frame...
            if highLossBottomLabel.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                highLossBottomLabel.frameNStart = frameN  # exact frame index
                highLossBottomLabel.tStart = t  # local t and not account for scr refresh
                highLossBottomLabel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(highLossBottomLabel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'highLossBottomLabel.started')
                # update status
                highLossBottomLabel.status = STARTED
                highLossBottomLabel.setAutoDraw(True)
            
            # if highLossBottomLabel is active this frame...
            if highLossBottomLabel.status == STARTED:
                # update params
                pass
            
            # *lowLossBottomLabel* updates
            
            # if lowLossBottomLabel is starting this frame...
            if lowLossBottomLabel.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lowLossBottomLabel.frameNStart = frameN  # exact frame index
                lowLossBottomLabel.tStart = t  # local t and not account for scr refresh
                lowLossBottomLabel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lowLossBottomLabel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lowLossBottomLabel.started')
                # update status
                lowLossBottomLabel.status = STARTED
                lowLossBottomLabel.setAutoDraw(True)
            
            # if lowLossBottomLabel is active this frame...
            if lowLossBottomLabel.status == STARTED:
                # update params
                pass
            
            # *highLossLabel* updates
            
            # if highLossLabel is starting this frame...
            if highLossLabel.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                highLossLabel.frameNStart = frameN  # exact frame index
                highLossLabel.tStart = t  # local t and not account for scr refresh
                highLossLabel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(highLossLabel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'highLossLabel.started')
                # update status
                highLossLabel.status = STARTED
                highLossLabel.setAutoDraw(True)
            
            # if highLossLabel is active this frame...
            if highLossLabel.status == STARTED:
                # update params
                pass
            
            # *lowLossLabel* updates
            
            # if lowLossLabel is starting this frame...
            if lowLossLabel.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lowLossLabel.frameNStart = frameN  # exact frame index
                lowLossLabel.tStart = t  # local t and not account for scr refresh
                lowLossLabel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lowLossLabel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lowLossLabel.started')
                # update status
                lowLossLabel.status = STARTED
                lowLossLabel.setAutoDraw(True)
            
            # if lowLossLabel is active this frame...
            if lowLossLabel.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=lossDirs,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                lossDirs.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in lossDirs.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "lossDirs" ---
        for thisComponent in lossDirs.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for lossDirs
        lossDirs.tStop = globalClock.getTime(format='float')
        lossDirs.tStopRefresh = tThisFlipGlobal
        thisExp.addData('lossDirs.stopped', lossDirs.tStop)
        # check responses
        if advanceScreen4.keys in ['', [], None]:  # No response was made
            advanceScreen4.keys = None
        pracBlock.addData('advanceScreen4.keys',advanceScreen4.keys)
        if advanceScreen4.keys != None:  # we had a response
            pracBlock.addData('advanceScreen4.rt', advanceScreen4.rt)
            pracBlock.addData('advanceScreen4.duration', advanceScreen4.duration)
        # Run 'End Routine' code from framePracOn
        frame.setAutoDraw(True)
        # the Routine "lossDirs" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        practice2 = data.TrialHandler2(
            name='practice2',
            nReps=pracOn, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('conditions/prac2Trials.csv'), 
            seed=None, 
        )
        thisExp.addLoop(practice2)  # add the loop to the experiment
        thisPractice2 = practice2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPractice2.rgb)
        if thisPractice2 != None:
            for paramName in thisPractice2:
                globals()[paramName] = thisPractice2[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPractice2 in practice2:
            practice2.status = STARTED
            if hasattr(thisPractice2, 'status'):
                thisPractice2.status = STARTED
            currentLoop = practice2
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPractice2.rgb)
            if thisPractice2 != None:
                for paramName in thisPractice2:
                    globals()[paramName] = thisPractice2[paramName]
            
            # --- Prepare to start Routine "prac2Cue" ---
            # create an object to store info about Routine prac2Cue
            prac2Cue = data.Routine(
                name='prac2Cue',
                components=[prac2LeftCue, prac2RightCue, prac2Resp, ISI],
            )
            prac2Cue.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for prac2Resp
            prac2Resp.keys = []
            prac2Resp.rt = []
            _prac2Resp_allKeys = []
            # allowedKeys looks like a variable, so make sure it exists locally
            if 'allowedKeys' in globals():
                allowedKeys = globals()['allowedKeys']
            # Run 'Begin Routine' code from setCuesPrac
            logging.info("Iterating through thisTrial")
            for value in practice2.trialList:
                logging.info(value)
            
            # update component parameters for each repeat
            optimalImg = cuePairPrac[condition]['optimalChoice']
            suboptimalImg = cuePairPrac[condition]['suboptimalChoice']
            currentLoop.addData('optimalImg', optimalImg)
            currentLoop.addData('suboptimalImg', suboptimalImg)
            
            
            # Assumes button box responses are '1' for index and '2' for middle finger.
            if np.random.uniform() > 0.5:
                cuePositions = {responseKeys['left']:optimalImg, responseKeys['right']:suboptimalImg}
            
            else:
                cuePositions =  {responseKeys['left']:suboptimalImg, responseKeys['right']:optimalImg}
                
            sidePositions = dict([(v, k) for (k, v) in cuePositions.items()])
            
            optimalSide='left' if sidePositions[optimalImg]==responseKeys['left'] else 'right'
            suboptimalSide='left' if sidePositions[suboptimalImg]==responseKeys['left'] else 'right'
            
            currentLoop.addData('optimalSide', optimalSide)
            currentLoop.addData('suboptimalSide', suboptimalSide)
            currentLoop.addData('optimalResponse',sidePositions[optimalImg])
            currentLoop.addData('suboptimalResponse',sidePositions[suboptimalImg])
            leftImage = 'stimuli_sets/'+cuePositions[responseKeys['left']]
            currentLoop.addData('leftImage', leftImage)
            rightImage = 'stimuli_sets/' +cuePositions[responseKeys['right']]
            currentLoop.addData('rightImage', rightImage)
            print(leftImage)
            print(rightImage)
            
            # Set the "correct" response by choosing the one that will give the better outcome.
            # That's a higher probability of winning in gain conditions and a lower probability 
            # of losing in loss conditions. Used for analysis, not for choosing outcomes.
            correct = sidePositions[optimalImg]
            # Run 'Begin Routine' code from pracSelect
            # Initialize cue end time for cues and selection indicator.
            # If no response is made in this time, the trial ends.
            # Otherwise, cueEndTime is updated to response
            # time plus the selection duration.
            cueEndTime = 3
            selectionIndicator.status = NOT_STARTED
            # Run 'Begin Routine' code from pracFrameLabels
            if condition == "lowgain":
                corrvalue=0.20
                incorrvalue=0
                topLabel="+$%.2f"%(abs(corrvalue))
                bottomLabel="+$%.2f"%(abs(incorrvalue))
            
            if condition == "highgain":
                corrvalue=1.00
                incorrvalue=0
                topLabel="+$%.2f"%(abs(corrvalue))
                bottomLabel="+$%.2f"%(abs(incorrvalue))
            
            if condition == "lowloss":
                corrvalue=0
                incorrvalue=-0.20
                topLabel="-$%.2f"%(abs(corrvalue))
                bottomLabel="-$%.2f"%(abs(incorrvalue))
            
            if condition == "highloss":
                corrvalue=0
                incorrvalue=-1.00
                topLabel="-$%.2f"%(abs(corrvalue))
                bottomLabel="-$%.2f"%(abs(incorrvalue))
                
            topFrameText.setText(topLabel)
            topFrameText.status = NOT_STARTED
            bottomFrameText.setText(bottomLabel)
            bottomFrameText.status = NOT_STARTED
            # store start times for prac2Cue
            prac2Cue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac2Cue.tStart = globalClock.getTime(format='float')
            prac2Cue.status = STARTED
            thisExp.addData('prac2Cue.started', prac2Cue.tStart)
            prac2Cue.maxDuration = None
            # keep track of which components have finished
            prac2CueComponents = prac2Cue.components
            for thisComponent in prac2Cue.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac2Cue" ---
            prac2Cue.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 3.0:
                # if trial has changed, end Routine now
                if hasattr(thisPractice2, 'status') and thisPractice2.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac2LeftCue* updates
                
                # if prac2LeftCue is starting this frame...
                if prac2LeftCue.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    prac2LeftCue.frameNStart = frameN  # exact frame index
                    prac2LeftCue.tStart = t  # local t and not account for scr refresh
                    prac2LeftCue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2LeftCue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2LeftCue.started')
                    # update status
                    prac2LeftCue.status = STARTED
                    prac2LeftCue.setAutoDraw(True)
                
                # if prac2LeftCue is active this frame...
                if prac2LeftCue.status == STARTED:
                    # update params
                    pass
                
                # if prac2LeftCue is stopping this frame...
                if prac2LeftCue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac2LeftCue.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        prac2LeftCue.tStop = t  # not accounting for scr refresh
                        prac2LeftCue.tStopRefresh = tThisFlipGlobal  # on global time
                        prac2LeftCue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac2LeftCue.stopped')
                        # update status
                        prac2LeftCue.status = FINISHED
                        prac2LeftCue.setAutoDraw(False)
                
                # *prac2RightCue* updates
                
                # if prac2RightCue is starting this frame...
                if prac2RightCue.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    prac2RightCue.frameNStart = frameN  # exact frame index
                    prac2RightCue.tStart = t  # local t and not account for scr refresh
                    prac2RightCue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2RightCue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2RightCue.started')
                    # update status
                    prac2RightCue.status = STARTED
                    prac2RightCue.setAutoDraw(True)
                
                # if prac2RightCue is active this frame...
                if prac2RightCue.status == STARTED:
                    # update params
                    pass
                
                # if prac2RightCue is stopping this frame...
                if prac2RightCue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac2RightCue.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        prac2RightCue.tStop = t  # not accounting for scr refresh
                        prac2RightCue.tStopRefresh = tThisFlipGlobal  # on global time
                        prac2RightCue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac2RightCue.stopped')
                        # update status
                        prac2RightCue.status = FINISHED
                        prac2RightCue.setAutoDraw(False)
                
                # *prac2Resp* updates
                waitOnFlip = False
                
                # if prac2Resp is starting this frame...
                if prac2Resp.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    prac2Resp.frameNStart = frameN  # exact frame index
                    prac2Resp.tStart = t  # local t and not account for scr refresh
                    prac2Resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2Resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2Resp.started')
                    # update status
                    prac2Resp.status = STARTED
                    # allowed keys looks like a variable named `allowedKeys`
                    if not type(allowedKeys) in [list, tuple, np.ndarray]:
                        if not isinstance(allowedKeys, str):
                            allowedKeys = str(allowedKeys)
                        elif not ',' in allowedKeys:
                            allowedKeys = (allowedKeys,)
                        else:
                            allowedKeys = eval(allowedKeys)
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(prac2Resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(prac2Resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if prac2Resp is stopping this frame...
                if prac2Resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac2Resp.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        prac2Resp.tStop = t  # not accounting for scr refresh
                        prac2Resp.tStopRefresh = tThisFlipGlobal  # on global time
                        prac2Resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac2Resp.stopped')
                        # update status
                        prac2Resp.status = FINISHED
                        prac2Resp.status = FINISHED
                if prac2Resp.status == STARTED and not waitOnFlip:
                    theseKeys = prac2Resp.getKeys(keyList=list(allowedKeys), ignoreKeys=["escape"], waitRelease=False)
                    _prac2Resp_allKeys.extend(theseKeys)
                    if len(_prac2Resp_allKeys):
                        prac2Resp.keys = _prac2Resp_allKeys[0].name  # just the first key pressed
                        prac2Resp.rt = _prac2Resp_allKeys[0].rt
                        prac2Resp.duration = _prac2Resp_allKeys[0].duration
                        # was this correct?
                        if (prac2Resp.keys == str(correct)) or (prac2Resp.keys == correct):
                            prac2Resp.corr = 1
                        else:
                            prac2Resp.corr = 0
                # Run 'Each Frame' code from pracSelect
                # If the selection has been drawn for the selection duration,
                # or the trial should end (cutting off the selection duration), end the routine
                if t > cueEndTime:
                    continueRoutine = False
                
                # Start Drawing Selection Indicator if Selection has been made
                if prac2Resp.keys and selectionIndicator.status == NOT_STARTED:
                    #cueComponents.append(selectionIndicator)
                    selectionIndicator.status = STARTED
                    selectionIndicator.setPos(selectionPosition(prac2Resp.keys))
                    selectionIndicator.setAutoDraw(True)
                    #if getSnapshots: win.getMovieFrame()
                    # Set End Time, clip to end of trial length (4 sec max)
                    # Don't worry about end of selection display in fMRI
                    #cueEndTime = t + selectionDuration
                # Run 'Each Frame' code from pracFrameLabels
                if t > .5 and topFrameText.status == NOT_STARTED:
                    topFrameText.setAutoDraw(True)
                    bottomFrameText.setAutoDraw(True)
                    topFrameText.status = STARTED
                # *ISI* period
                
                # if ISI is starting this frame...
                if ISI.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    ISI.frameNStart = frameN  # exact frame index
                    ISI.tStart = t  # local t and not account for scr refresh
                    ISI.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ISI, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('ISI.started', t)
                    # update status
                    ISI.status = STARTED
                    # start the static period
                    ISI.start(0.5)
                elif ISI.status == STARTED:  # one frame should pass before updating params and completing
                    # Updating other components during *ISI*
                    prac2LeftCue.setImage(leftImage)
                    prac2RightCue.setImage(rightImage)
                    # Component updates done
                    # finish the static period and store its duration
                    ISI.complete()
                    ISI.tStop = ISI.tStart + ISI.getDuration()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac2Cue,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac2Cue.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac2Cue.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac2Cue" ---
            for thisComponent in prac2Cue.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac2Cue
            prac2Cue.tStop = globalClock.getTime(format='float')
            prac2Cue.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac2Cue.stopped', prac2Cue.tStop)
            # check responses
            if prac2Resp.keys in ['', [], None]:  # No response was made
                prac2Resp.keys = None
                # was no response the correct answer?!
                if str(correct).lower() == 'none':
                   prac2Resp.corr = 1;  # correct non-response
                else:
                   prac2Resp.corr = 0;  # failed to respond (incorrectly)
            # store data for practice2 (TrialHandler)
            practice2.addData('prac2Resp.keys',prac2Resp.keys)
            practice2.addData('prac2Resp.corr', prac2Resp.corr)
            if prac2Resp.keys != None:  # we had a response
                practice2.addData('prac2Resp.rt', prac2Resp.rt)
                practice2.addData('prac2Resp.duration', prac2Resp.duration)
            # Run 'End Routine' code from setCuesPrac
            if prac2Resp.keys == responseKeys['left']:
                currentLoop.addData('optedSide', 'left')
                currentLoop.addData('optedImg', leftImage)
                
            elif prac2Resp.keys == responseKeys['right']:
                currentLoop.addData('optedSide', 'right')
                currentLoop.addData('optedImg', rightImage)
                
            else: 
                currentLoop.addData('optedSide', 'NA')
                currentLoop.addData('optedImg', 'NA')
            # Run 'End Routine' code from pracSelect
            selectionIndicator.setAutoDraw(False)
            # Run 'End Routine' code from pracFrameLabels
            topFrameText.status = NOT_STARTED
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if prac2Cue.maxDurationReached:
                routineTimer.addTime(-prac2Cue.maxDuration)
            elif prac2Cue.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)
            
            # --- Prepare to start Routine "prac2Jitter" ---
            # create an object to store info about Routine prac2Jitter
            prac2Jitter = data.Routine(
                name='prac2Jitter',
                components=[prac2Fix],
            )
            prac2Jitter.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for prac2Jitter
            prac2Jitter.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac2Jitter.tStart = globalClock.getTime(format='float')
            prac2Jitter.status = STARTED
            thisExp.addData('prac2Jitter.started', prac2Jitter.tStart)
            prac2Jitter.maxDuration = None
            # keep track of which components have finished
            prac2JitterComponents = prac2Jitter.components
            for thisComponent in prac2Jitter.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac2Jitter" ---
            prac2Jitter.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPractice2, 'status') and thisPractice2.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prac2Fix* updates
                
                # if prac2Fix is starting this frame...
                if prac2Fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prac2Fix.frameNStart = frameN  # exact frame index
                    prac2Fix.tStart = t  # local t and not account for scr refresh
                    prac2Fix.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prac2Fix, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac2Fix.started')
                    # update status
                    prac2Fix.status = STARTED
                    prac2Fix.setAutoDraw(True)
                
                # if prac2Fix is active this frame...
                if prac2Fix.status == STARTED:
                    # update params
                    pass
                
                # if prac2Fix is stopping this frame...
                if prac2Fix.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prac2Fix.tStartRefresh + outcomeDelay-frameTolerance:
                        # keep track of stop time/frame for later
                        prac2Fix.tStop = t  # not accounting for scr refresh
                        prac2Fix.tStopRefresh = tThisFlipGlobal  # on global time
                        prac2Fix.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prac2Fix.stopped')
                        # update status
                        prac2Fix.status = FINISHED
                        prac2Fix.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac2Jitter,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac2Jitter.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac2Jitter.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac2Jitter" ---
            for thisComponent in prac2Jitter.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac2Jitter
            prac2Jitter.tStop = globalClock.getTime(format='float')
            prac2Jitter.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac2Jitter.stopped', prac2Jitter.tStop)
            # the Routine "prac2Jitter" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "prac2Out" ---
            # create an object to store info about Routine prac2Out
            prac2Out = data.Routine(
                name='prac2Out',
                components=[outcomePracText],
            )
            prac2Out.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from pracRightChoice
            if prac2Resp.keys==sidePositions[optimalImg]:
                Choice="optimal"
            elif prac2Resp.keys==sidePositions[suboptimalImg]:
                Choice="suboptimal"
            else: 
                Choice="NA"
                
            Accuracy=prac2Resp.corr
            
            currentLoop.addData('optedFor', Choice)
            currentLoop.addData('accuracy', Accuracy)
            # Run 'Begin Routine' code from definePracOut
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
            if prac2Resp.keys:
                outcome = choose_prac_outcome_from_cue(prac2Resp)
                currentLoop.addData('outcome', outcome)
                if outcome:
                    # If the outcome was true, display the change. 
                    # Format prettily using the operator instead of just using the change value.
                    change = corrvalue
                    outcome_msg = "%s$%.2f\nCorrect"%(trueOperator, abs(change))
                    outcome_image = None
                    Feedback=1
            
                else: 
                    # If outcome was False, the trial doesn't change anything.
                    # Still display the operator for trial condition
                    change = incorrvalue
                    outcome_msg = "%s$%.2f\nIncorrect" % (falseOperator,abs(change))
                    outcome_image = None
                    Feedback=0
            else: # give incorrect outcome for missed trials
                change = incorrvalue
                currentLoop.addData('outcome', None)
                outcome_msg = "%s$%.2f\nMissed" % (falseOperator,abs(change))
                outcome_image = None
                Feedback="NA"
            
            
            
            # Add a positive or negative change
            bank += change
            
            currentLoop.addData('Feedback', Feedback)
            currentLoop.addData('gainLossAmount', change)
            currentLoop.addData('runningBankTotal', bank)
            currentLoop.addData('feedbackTime', fmriClock.getTime())
            
            outcomePracText.setText(outcome_msg)
            # store start times for prac2Out
            prac2Out.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            prac2Out.tStart = globalClock.getTime(format='float')
            prac2Out.status = STARTED
            thisExp.addData('prac2Out.started', prac2Out.tStart)
            prac2Out.maxDuration = None
            # keep track of which components have finished
            prac2OutComponents = prac2Out.components
            for thisComponent in prac2Out.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "prac2Out" ---
            prac2Out.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # if trial has changed, end Routine now
                if hasattr(thisPractice2, 'status') and thisPractice2.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *outcomePracText* updates
                
                # if outcomePracText is starting this frame...
                if outcomePracText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    outcomePracText.frameNStart = frameN  # exact frame index
                    outcomePracText.tStart = t  # local t and not account for scr refresh
                    outcomePracText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(outcomePracText, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'outcomePracText.started')
                    # update status
                    outcomePracText.status = STARTED
                    outcomePracText.setAutoDraw(True)
                
                # if outcomePracText is active this frame...
                if outcomePracText.status == STARTED:
                    # update params
                    pass
                
                # if outcomePracText is stopping this frame...
                if outcomePracText.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > outcomePracText.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        outcomePracText.tStop = t  # not accounting for scr refresh
                        outcomePracText.tStopRefresh = tThisFlipGlobal  # on global time
                        outcomePracText.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'outcomePracText.stopped')
                        # update status
                        outcomePracText.status = FINISHED
                        outcomePracText.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=prac2Out,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    prac2Out.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in prac2Out.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "prac2Out" ---
            for thisComponent in prac2Out.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for prac2Out
            prac2Out.tStop = globalClock.getTime(format='float')
            prac2Out.tStopRefresh = tThisFlipGlobal
            thisExp.addData('prac2Out.stopped', prac2Out.tStop)
            # Run 'End Routine' code from definePracOut
            expInfo['bank'] = bank
            currentLoop.addData('feedbackOffTime', fmriClock.getTime())
            # Run 'End Routine' code from frameOff
            topFrameText.setAutoDraw(False)
            bottomFrameText.setAutoDraw(False)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if prac2Out.maxDurationReached:
                routineTimer.addTime(-prac2Out.maxDuration)
            elif prac2Out.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            # mark thisPractice2 as finished
            if hasattr(thisPractice2, 'status'):
                thisPractice2.status = FINISHED
            # if awaiting a pause, pause now
            if practice2.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                practice2.status = STARTED
            thisExp.nextEntry()
            
        # completed pracOn repeats of 'practice2'
        practice2.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if practice2.trialList in ([], [None], None):
            params = []
        else:
            params = practice2.trialList[0].keys()
        # save data for this loop
        practice2.saveAsText(filename + '_practice2.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "check3" ---
        # create an object to store info about Routine check3
        check3 = data.Routine(
            name='check3',
            components=[check3Text, check3ImageLeft, check3ImageRight, check3Press, down2],
        )
        check3.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from check3Select
        # Initialize cue end time for cues and selection indicator.
        # If no response is made in this time, the trial ends.
        # Otherwise, cueEndTime is updated to response
        # time plus the selection duration.
        frame.setAutoDraw(False)
        selectionIndicatorPrac.status = NOT_STARTED
        t_started = False
        # create starting attributes for check3Press
        check3Press.keys = []
        check3Press.rt = []
        _check3Press_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'allowedKeys' in globals():
            allowedKeys = globals()['allowedKeys']
        # create starting attributes for down2
        down2.keys = []
        down2.rt = []
        _down2_allKeys = []
        # store start times for check3
        check3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        check3.tStart = globalClock.getTime(format='float')
        check3.status = STARTED
        thisExp.addData('check3.started', check3.tStart)
        check3.maxDuration = None
        # keep track of which components have finished
        check3Components = check3.components
        for thisComponent in check3.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "check3" ---
        check3.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPracBlock, 'status') and thisPracBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *check3Text* updates
            
            # if check3Text is starting this frame...
            if check3Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check3Text.frameNStart = frameN  # exact frame index
                check3Text.tStart = t  # local t and not account for scr refresh
                check3Text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check3Text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check3Text.started')
                # update status
                check3Text.status = STARTED
                check3Text.setAutoDraw(True)
            
            # if check3Text is active this frame...
            if check3Text.status == STARTED:
                # update params
                pass
            
            # *check3ImageLeft* updates
            
            # if check3ImageLeft is starting this frame...
            if check3ImageLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check3ImageLeft.frameNStart = frameN  # exact frame index
                check3ImageLeft.tStart = t  # local t and not account for scr refresh
                check3ImageLeft.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check3ImageLeft, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check3ImageLeft.started')
                # update status
                check3ImageLeft.status = STARTED
                check3ImageLeft.setAutoDraw(True)
            
            # if check3ImageLeft is active this frame...
            if check3ImageLeft.status == STARTED:
                # update params
                pass
            
            # *check3ImageRight* updates
            
            # if check3ImageRight is starting this frame...
            if check3ImageRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check3ImageRight.frameNStart = frameN  # exact frame index
                check3ImageRight.tStart = t  # local t and not account for scr refresh
                check3ImageRight.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check3ImageRight, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check3ImageRight.started')
                # update status
                check3ImageRight.status = STARTED
                check3ImageRight.setAutoDraw(True)
            
            # if check3ImageRight is active this frame...
            if check3ImageRight.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from check3Select
            # If the selection has been drawn for the selection duration,
            # or the trial should end (cutting off the selection duration), end the routine
            if t_started:
                if t > t_started + 5:
                    continueRoutine = False
            
            # Start Drawing Selection Indicator if Selection has been made
            if check3Press.keys and selectionIndicatorPrac.status == NOT_STARTED:
                #cueComponents.append(selectionIndicatorPrac)
                selectionIndicatorPrac.status = STARTED
                selectionIndicatorPrac.setPos(selectionPosition(check3Press.keys))
                selectionIndicatorPrac.setAutoDraw(True)
                t_started = core.Clock().getTime()
            
            # *check3Press* updates
            waitOnFlip = False
            
            # if check3Press is starting this frame...
            if check3Press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check3Press.frameNStart = frameN  # exact frame index
                check3Press.tStart = t  # local t and not account for scr refresh
                check3Press.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check3Press, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check3Press.started')
                # update status
                check3Press.status = STARTED
                # allowed keys looks like a variable named `allowedKeys`
                if not type(allowedKeys) in [list, tuple, np.ndarray]:
                    if not isinstance(allowedKeys, str):
                        allowedKeys = str(allowedKeys)
                    elif not ',' in allowedKeys:
                        allowedKeys = (allowedKeys,)
                    else:
                        allowedKeys = eval(allowedKeys)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(check3Press.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(check3Press.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if check3Press.status == STARTED and not waitOnFlip:
                theseKeys = check3Press.getKeys(keyList=list(allowedKeys), ignoreKeys=["escape"], waitRelease=False)
                _check3Press_allKeys.extend(theseKeys)
                if len(_check3Press_allKeys):
                    check3Press.keys = _check3Press_allKeys[-1].name  # just the last key pressed
                    check3Press.rt = _check3Press_allKeys[-1].rt
                    check3Press.duration = _check3Press_allKeys[-1].duration
            
            # *down2* updates
            waitOnFlip = False
            
            # if down2 is starting this frame...
            if down2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                down2.frameNStart = frameN  # exact frame index
                down2.tStart = t  # local t and not account for scr refresh
                down2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(down2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'down2.started')
                # update status
                down2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(down2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(down2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if down2.status == STARTED and not waitOnFlip:
                theseKeys = down2.getKeys(keyList=['down'], ignoreKeys=["escape"], waitRelease=False)
                _down2_allKeys.extend(theseKeys)
                if len(_down2_allKeys):
                    down2.keys = _down2_allKeys[-1].name  # just the last key pressed
                    down2.rt = _down2_allKeys[-1].rt
                    down2.duration = _down2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=check3,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                check3.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in check3.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "check3" ---
        for thisComponent in check3.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for check3
        check3.tStop = globalClock.getTime(format='float')
        check3.tStopRefresh = tThisFlipGlobal
        thisExp.addData('check3.stopped', check3.tStop)
        # Run 'End Routine' code from check3Select
        selectionIndicatorPrac.setAutoDraw(False)
        
        # check responses
        if check3Press.keys in ['', [], None]:  # No response was made
            check3Press.keys = None
        pracBlock.addData('check3Press.keys',check3Press.keys)
        if check3Press.keys != None:  # we had a response
            pracBlock.addData('check3Press.rt', check3Press.rt)
            pracBlock.addData('check3Press.duration', check3Press.duration)
        # check responses
        if down2.keys in ['', [], None]:  # No response was made
            down2.keys = None
        pracBlock.addData('down2.keys',down2.keys)
        if down2.keys != None:  # we had a response
            pracBlock.addData('down2.rt', down2.rt)
            pracBlock.addData('down2.duration', down2.duration)
        # the Routine "check3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "check2" ---
        # create an object to store info about Routine check2
        check2 = data.Routine(
            name='check2',
            components=[check2Text, check2ImageLeft, check2ImageRight, check2Press, down3],
        )
        check2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from check2Select
        # Initialize cue end time for cues and selection indicator.
        # If no response is made in this time, the trial ends.
        # Otherwise, cueEndTime is updated to response
        # time plus the selection duration.
        
        selectionIndicatorPrac.status = NOT_STARTED
        t_started = False
        # create starting attributes for check2Press
        check2Press.keys = []
        check2Press.rt = []
        _check2Press_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'allowedKeys' in globals():
            allowedKeys = globals()['allowedKeys']
        # create starting attributes for down3
        down3.keys = []
        down3.rt = []
        _down3_allKeys = []
        # store start times for check2
        check2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        check2.tStart = globalClock.getTime(format='float')
        check2.status = STARTED
        thisExp.addData('check2.started', check2.tStart)
        check2.maxDuration = None
        # keep track of which components have finished
        check2Components = check2.components
        for thisComponent in check2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "check2" ---
        check2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPracBlock, 'status') and thisPracBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *check2Text* updates
            
            # if check2Text is starting this frame...
            if check2Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check2Text.frameNStart = frameN  # exact frame index
                check2Text.tStart = t  # local t and not account for scr refresh
                check2Text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check2Text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check2Text.started')
                # update status
                check2Text.status = STARTED
                check2Text.setAutoDraw(True)
            
            # if check2Text is active this frame...
            if check2Text.status == STARTED:
                # update params
                pass
            
            # *check2ImageLeft* updates
            
            # if check2ImageLeft is starting this frame...
            if check2ImageLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check2ImageLeft.frameNStart = frameN  # exact frame index
                check2ImageLeft.tStart = t  # local t and not account for scr refresh
                check2ImageLeft.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check2ImageLeft, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check2ImageLeft.started')
                # update status
                check2ImageLeft.status = STARTED
                check2ImageLeft.setAutoDraw(True)
            
            # if check2ImageLeft is active this frame...
            if check2ImageLeft.status == STARTED:
                # update params
                pass
            
            # *check2ImageRight* updates
            
            # if check2ImageRight is starting this frame...
            if check2ImageRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check2ImageRight.frameNStart = frameN  # exact frame index
                check2ImageRight.tStart = t  # local t and not account for scr refresh
                check2ImageRight.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check2ImageRight, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check2ImageRight.started')
                # update status
                check2ImageRight.status = STARTED
                check2ImageRight.setAutoDraw(True)
            
            # if check2ImageRight is active this frame...
            if check2ImageRight.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from check2Select
            # If the selection has been drawn for the selection duration,
            # or the trial should end (cutting off the selection duration), end the routine
            if t_started:
                if t > t_started + 5:
                    continueRoutine = False
            
            # Start Drawing Selection Indicator if Selection has been made
            if check2Press.keys and selectionIndicatorPrac.status == NOT_STARTED:
                #cueComponents.append(selectionIndicatorPrac)
                selectionIndicatorPrac.status = STARTED
                selectionIndicatorPrac.setPos(selectionPosition(check2Press.keys))
                selectionIndicatorPrac.setAutoDraw(True)
                t_started = core.Clock().getTime()
            
            # *check2Press* updates
            waitOnFlip = False
            
            # if check2Press is starting this frame...
            if check2Press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check2Press.frameNStart = frameN  # exact frame index
                check2Press.tStart = t  # local t and not account for scr refresh
                check2Press.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check2Press, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check2Press.started')
                # update status
                check2Press.status = STARTED
                # allowed keys looks like a variable named `allowedKeys`
                if not type(allowedKeys) in [list, tuple, np.ndarray]:
                    if not isinstance(allowedKeys, str):
                        allowedKeys = str(allowedKeys)
                    elif not ',' in allowedKeys:
                        allowedKeys = (allowedKeys,)
                    else:
                        allowedKeys = eval(allowedKeys)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(check2Press.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(check2Press.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if check2Press.status == STARTED and not waitOnFlip:
                theseKeys = check2Press.getKeys(keyList=list(allowedKeys), ignoreKeys=["escape"], waitRelease=False)
                _check2Press_allKeys.extend(theseKeys)
                if len(_check2Press_allKeys):
                    check2Press.keys = _check2Press_allKeys[-1].name  # just the last key pressed
                    check2Press.rt = _check2Press_allKeys[-1].rt
                    check2Press.duration = _check2Press_allKeys[-1].duration
            
            # *down3* updates
            waitOnFlip = False
            
            # if down3 is starting this frame...
            if down3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                down3.frameNStart = frameN  # exact frame index
                down3.tStart = t  # local t and not account for scr refresh
                down3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(down3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'down3.started')
                # update status
                down3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(down3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(down3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if down3.status == STARTED and not waitOnFlip:
                theseKeys = down3.getKeys(keyList=['down'], ignoreKeys=["escape"], waitRelease=False)
                _down3_allKeys.extend(theseKeys)
                if len(_down3_allKeys):
                    down3.keys = _down3_allKeys[-1].name  # just the last key pressed
                    down3.rt = _down3_allKeys[-1].rt
                    down3.duration = _down3_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=check2,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                check2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in check2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "check2" ---
        for thisComponent in check2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for check2
        check2.tStop = globalClock.getTime(format='float')
        check2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('check2.stopped', check2.tStop)
        # Run 'End Routine' code from check2Select
        selectionIndicatorPrac.setAutoDraw(False)
        
        # check responses
        if check2Press.keys in ['', [], None]:  # No response was made
            check2Press.keys = None
        pracBlock.addData('check2Press.keys',check2Press.keys)
        if check2Press.keys != None:  # we had a response
            pracBlock.addData('check2Press.rt', check2Press.rt)
            pracBlock.addData('check2Press.duration', check2Press.duration)
        # check responses
        if down3.keys in ['', [], None]:  # No response was made
            down3.keys = None
        pracBlock.addData('down3.keys',down3.keys)
        if down3.keys != None:  # we had a response
            pracBlock.addData('down3.rt', down3.rt)
            pracBlock.addData('down3.duration', down3.duration)
        # the Routine "check2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "check4" ---
        # create an object to store info about Routine check4
        check4 = data.Routine(
            name='check4',
            components=[check4Text, check4ImageLeft, check4ImageRight, check4Press, down4],
        )
        check4.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from check4Select
        # Initialize cue end time for cues and selection indicator.
        # If no response is made in this time, the trial ends.
        # Otherwise, cueEndTime is updated to response
        # time plus the selection duration.
        
        selectionIndicatorPrac.status = NOT_STARTED
        t_started = False
        # create starting attributes for check4Press
        check4Press.keys = []
        check4Press.rt = []
        _check4Press_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'allowedKeys' in globals():
            allowedKeys = globals()['allowedKeys']
        # create starting attributes for down4
        down4.keys = []
        down4.rt = []
        _down4_allKeys = []
        # store start times for check4
        check4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        check4.tStart = globalClock.getTime(format='float')
        check4.status = STARTED
        thisExp.addData('check4.started', check4.tStart)
        check4.maxDuration = None
        # keep track of which components have finished
        check4Components = check4.components
        for thisComponent in check4.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "check4" ---
        check4.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPracBlock, 'status') and thisPracBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *check4Text* updates
            
            # if check4Text is starting this frame...
            if check4Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check4Text.frameNStart = frameN  # exact frame index
                check4Text.tStart = t  # local t and not account for scr refresh
                check4Text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check4Text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check4Text.started')
                # update status
                check4Text.status = STARTED
                check4Text.setAutoDraw(True)
            
            # if check4Text is active this frame...
            if check4Text.status == STARTED:
                # update params
                pass
            
            # *check4ImageLeft* updates
            
            # if check4ImageLeft is starting this frame...
            if check4ImageLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check4ImageLeft.frameNStart = frameN  # exact frame index
                check4ImageLeft.tStart = t  # local t and not account for scr refresh
                check4ImageLeft.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check4ImageLeft, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check4ImageLeft.started')
                # update status
                check4ImageLeft.status = STARTED
                check4ImageLeft.setAutoDraw(True)
            
            # if check4ImageLeft is active this frame...
            if check4ImageLeft.status == STARTED:
                # update params
                pass
            
            # *check4ImageRight* updates
            
            # if check4ImageRight is starting this frame...
            if check4ImageRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check4ImageRight.frameNStart = frameN  # exact frame index
                check4ImageRight.tStart = t  # local t and not account for scr refresh
                check4ImageRight.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check4ImageRight, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check4ImageRight.started')
                # update status
                check4ImageRight.status = STARTED
                check4ImageRight.setAutoDraw(True)
            
            # if check4ImageRight is active this frame...
            if check4ImageRight.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from check4Select
            # If the selection has been drawn for the selection duration,
            # or the trial should end (cutting off the selection duration), end the routine
            if t_started:
                if t > t_started + 5:
                    continueRoutine = False
            
            # Start Drawing Selection Indicator if Selection has been made
            if check4Press.keys and selectionIndicatorPrac.status == NOT_STARTED:
                #cueComponents.append(selectionIndicatorPrac)
                selectionIndicatorPrac.status = STARTED
                selectionIndicatorPrac.setPos(selectionPosition(check4Press.keys))
                selectionIndicatorPrac.setAutoDraw(True)
                t_started = core.Clock().getTime()
            
            # *check4Press* updates
            waitOnFlip = False
            
            # if check4Press is starting this frame...
            if check4Press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check4Press.frameNStart = frameN  # exact frame index
                check4Press.tStart = t  # local t and not account for scr refresh
                check4Press.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check4Press, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check4Press.started')
                # update status
                check4Press.status = STARTED
                # allowed keys looks like a variable named `allowedKeys`
                if not type(allowedKeys) in [list, tuple, np.ndarray]:
                    if not isinstance(allowedKeys, str):
                        allowedKeys = str(allowedKeys)
                    elif not ',' in allowedKeys:
                        allowedKeys = (allowedKeys,)
                    else:
                        allowedKeys = eval(allowedKeys)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(check4Press.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(check4Press.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if check4Press.status == STARTED and not waitOnFlip:
                theseKeys = check4Press.getKeys(keyList=list(allowedKeys), ignoreKeys=["escape"], waitRelease=False)
                _check4Press_allKeys.extend(theseKeys)
                if len(_check4Press_allKeys):
                    check4Press.keys = _check4Press_allKeys[-1].name  # just the last key pressed
                    check4Press.rt = _check4Press_allKeys[-1].rt
                    check4Press.duration = _check4Press_allKeys[-1].duration
            
            # *down4* updates
            waitOnFlip = False
            
            # if down4 is starting this frame...
            if down4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                down4.frameNStart = frameN  # exact frame index
                down4.tStart = t  # local t and not account for scr refresh
                down4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(down4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'down4.started')
                # update status
                down4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(down4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(down4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if down4.status == STARTED and not waitOnFlip:
                theseKeys = down4.getKeys(keyList=['down'], ignoreKeys=["escape"], waitRelease=False)
                _down4_allKeys.extend(theseKeys)
                if len(_down4_allKeys):
                    down4.keys = _down4_allKeys[-1].name  # just the last key pressed
                    down4.rt = _down4_allKeys[-1].rt
                    down4.duration = _down4_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=check4,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                check4.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in check4.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "check4" ---
        for thisComponent in check4.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for check4
        check4.tStop = globalClock.getTime(format='float')
        check4.tStopRefresh = tThisFlipGlobal
        thisExp.addData('check4.stopped', check4.tStop)
        # Run 'End Routine' code from check4Select
        selectionIndicatorPrac.setAutoDraw(False)
        
        # check responses
        if check4Press.keys in ['', [], None]:  # No response was made
            check4Press.keys = None
        pracBlock.addData('check4Press.keys',check4Press.keys)
        if check4Press.keys != None:  # we had a response
            pracBlock.addData('check4Press.rt', check4Press.rt)
            pracBlock.addData('check4Press.duration', check4Press.duration)
        # check responses
        if down4.keys in ['', [], None]:  # No response was made
            down4.keys = None
        pracBlock.addData('down4.keys',down4.keys)
        if down4.keys != None:  # we had a response
            pracBlock.addData('down4.rt', down4.rt)
            pracBlock.addData('down4.duration', down4.duration)
        # the Routine "check4" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "check5" ---
        # create an object to store info about Routine check5
        check5 = data.Routine(
            name='check5',
            components=[check5Text, check5ImageLeft, check5ImageRight, check5Press, down5],
        )
        check5.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from check5Select
        # Initialize cue end time for cues and selection indicator.
        # If no response is made in this time, the trial ends.
        # Otherwise, cueEndTime is updated to response
        # time plus the selection duration.
        
        selectionIndicatorPrac.status = NOT_STARTED
        t_started = False
        # create starting attributes for check5Press
        check5Press.keys = []
        check5Press.rt = []
        _check5Press_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'allowedKeys' in globals():
            allowedKeys = globals()['allowedKeys']
        # create starting attributes for down5
        down5.keys = []
        down5.rt = []
        _down5_allKeys = []
        # store start times for check5
        check5.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        check5.tStart = globalClock.getTime(format='float')
        check5.status = STARTED
        thisExp.addData('check5.started', check5.tStart)
        check5.maxDuration = None
        # keep track of which components have finished
        check5Components = check5.components
        for thisComponent in check5.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "check5" ---
        check5.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPracBlock, 'status') and thisPracBlock.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *check5Text* updates
            
            # if check5Text is starting this frame...
            if check5Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check5Text.frameNStart = frameN  # exact frame index
                check5Text.tStart = t  # local t and not account for scr refresh
                check5Text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check5Text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check5Text.started')
                # update status
                check5Text.status = STARTED
                check5Text.setAutoDraw(True)
            
            # if check5Text is active this frame...
            if check5Text.status == STARTED:
                # update params
                pass
            
            # *check5ImageLeft* updates
            
            # if check5ImageLeft is starting this frame...
            if check5ImageLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check5ImageLeft.frameNStart = frameN  # exact frame index
                check5ImageLeft.tStart = t  # local t and not account for scr refresh
                check5ImageLeft.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check5ImageLeft, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check5ImageLeft.started')
                # update status
                check5ImageLeft.status = STARTED
                check5ImageLeft.setAutoDraw(True)
            
            # if check5ImageLeft is active this frame...
            if check5ImageLeft.status == STARTED:
                # update params
                pass
            
            # *check5ImageRight* updates
            
            # if check5ImageRight is starting this frame...
            if check5ImageRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check5ImageRight.frameNStart = frameN  # exact frame index
                check5ImageRight.tStart = t  # local t and not account for scr refresh
                check5ImageRight.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check5ImageRight, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check5ImageRight.started')
                # update status
                check5ImageRight.status = STARTED
                check5ImageRight.setAutoDraw(True)
            
            # if check5ImageRight is active this frame...
            if check5ImageRight.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from check5Select
            # If the selection has been drawn for the selection duration,
            # or the trial should end (cutting off the selection duration), end the routine
            if t_started:
                if t > t_started + 5:
                    continueRoutine = False
            
            # Start Drawing Selection Indicator if Selection has been made
            if check5Press.keys and selectionIndicatorPrac.status == NOT_STARTED:
                #cueComponents.append(selectionIndicatorPrac)
                selectionIndicatorPrac.status = STARTED
                selectionIndicatorPrac.setPos(selectionPosition(check5Press.keys))
                selectionIndicatorPrac.setAutoDraw(True)
                t_started = core.Clock().getTime()
            
            # *check5Press* updates
            waitOnFlip = False
            
            # if check5Press is starting this frame...
            if check5Press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                check5Press.frameNStart = frameN  # exact frame index
                check5Press.tStart = t  # local t and not account for scr refresh
                check5Press.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(check5Press, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'check5Press.started')
                # update status
                check5Press.status = STARTED
                # allowed keys looks like a variable named `allowedKeys`
                if not type(allowedKeys) in [list, tuple, np.ndarray]:
                    if not isinstance(allowedKeys, str):
                        allowedKeys = str(allowedKeys)
                    elif not ',' in allowedKeys:
                        allowedKeys = (allowedKeys,)
                    else:
                        allowedKeys = eval(allowedKeys)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(check5Press.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(check5Press.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if check5Press.status == STARTED and not waitOnFlip:
                theseKeys = check5Press.getKeys(keyList=list(allowedKeys), ignoreKeys=["escape"], waitRelease=False)
                _check5Press_allKeys.extend(theseKeys)
                if len(_check5Press_allKeys):
                    check5Press.keys = _check5Press_allKeys[-1].name  # just the last key pressed
                    check5Press.rt = _check5Press_allKeys[-1].rt
                    check5Press.duration = _check5Press_allKeys[-1].duration
            
            # *down5* updates
            waitOnFlip = False
            
            # if down5 is starting this frame...
            if down5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                down5.frameNStart = frameN  # exact frame index
                down5.tStart = t  # local t and not account for scr refresh
                down5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(down5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'down5.started')
                # update status
                down5.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(down5.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(down5.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if down5.status == STARTED and not waitOnFlip:
                theseKeys = down5.getKeys(keyList=['down'], ignoreKeys=["escape"], waitRelease=False)
                _down5_allKeys.extend(theseKeys)
                if len(_down5_allKeys):
                    down5.keys = _down5_allKeys[-1].name  # just the last key pressed
                    down5.rt = _down5_allKeys[-1].rt
                    down5.duration = _down5_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=check5,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                check5.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in check5.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "check5" ---
        for thisComponent in check5.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for check5
        check5.tStop = globalClock.getTime(format='float')
        check5.tStopRefresh = tThisFlipGlobal
        thisExp.addData('check5.stopped', check5.tStop)
        # Run 'End Routine' code from check5Select
        selectionIndicatorPrac.setAutoDraw(False)
        
        # check responses
        if check5Press.keys in ['', [], None]:  # No response was made
            check5Press.keys = None
        pracBlock.addData('check5Press.keys',check5Press.keys)
        if check5Press.keys != None:  # we had a response
            pracBlock.addData('check5Press.rt', check5Press.rt)
            pracBlock.addData('check5Press.duration', check5Press.duration)
        # check responses
        if down5.keys in ['', [], None]:  # No response was made
            down5.keys = None
        pracBlock.addData('down5.keys',down5.keys)
        if down5.keys != None:  # we had a response
            pracBlock.addData('down5.rt', down5.rt)
            pracBlock.addData('down5.duration', down5.duration)
        # the Routine "check5" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisPracBlock as finished
        if hasattr(thisPracBlock, 'status'):
            thisPracBlock.status = FINISHED
        # if awaiting a pause, pause now
        if pracBlock.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            pracBlock.status = STARTED
        thisExp.nextEntry()
        
    # completed pracBlockRepeats repeats of 'pracBlock'
    pracBlock.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if pracBlock.trialList in ([], [None], None):
        params = []
    else:
        params = pracBlock.trialList[0].keys()
    # save data for this loop
    pracBlock.saveAsText(filename + '_pracBlock.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "summaryDirections" ---
    # create an object to store info about Routine summaryDirections
    summaryDirections = data.Routine(
        name='summaryDirections',
        components=[summary, advanceScreenPress4],
    )
    summaryDirections.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for advanceScreenPress4
    advanceScreenPress4.keys = []
    advanceScreenPress4.rt = []
    _advanceScreenPress4_allKeys = []
    # Run 'Begin Routine' code from summaryLeftCode
    summary.alignText = 'left'
    # store start times for summaryDirections
    summaryDirections.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    summaryDirections.tStart = globalClock.getTime(format='float')
    summaryDirections.status = STARTED
    thisExp.addData('summaryDirections.started', summaryDirections.tStart)
    summaryDirections.maxDuration = None
    # keep track of which components have finished
    summaryDirectionsComponents = summaryDirections.components
    for thisComponent in summaryDirections.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "summaryDirections" ---
    summaryDirections.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *summary* updates
        
        # if summary is starting this frame...
        if summary.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            summary.frameNStart = frameN  # exact frame index
            summary.tStart = t  # local t and not account for scr refresh
            summary.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(summary, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'summary.started')
            # update status
            summary.status = STARTED
            summary.setAutoDraw(True)
        
        # if summary is active this frame...
        if summary.status == STARTED:
            # update params
            pass
        
        # *advanceScreenPress4* updates
        waitOnFlip = False
        
        # if advanceScreenPress4 is starting this frame...
        if advanceScreenPress4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreenPress4.frameNStart = frameN  # exact frame index
            advanceScreenPress4.tStart = t  # local t and not account for scr refresh
            advanceScreenPress4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreenPress4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreenPress4.started')
            # update status
            advanceScreenPress4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreenPress4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreenPress4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreenPress4.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreenPress4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _advanceScreenPress4_allKeys.extend(theseKeys)
            if len(_advanceScreenPress4_allKeys):
                advanceScreenPress4.keys = _advanceScreenPress4_allKeys[-1].name  # just the last key pressed
                advanceScreenPress4.rt = _advanceScreenPress4_allKeys[-1].rt
                advanceScreenPress4.duration = _advanceScreenPress4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=summaryDirections,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            summaryDirections.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in summaryDirections.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "summaryDirections" ---
    for thisComponent in summaryDirections.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for summaryDirections
    summaryDirections.tStop = globalClock.getTime(format='float')
    summaryDirections.tStopRefresh = tThisFlipGlobal
    thisExp.addData('summaryDirections.stopped', summaryDirections.tStop)
    # check responses
    if advanceScreenPress4.keys in ['', [], None]:  # No response was made
        advanceScreenPress4.keys = None
    thisExp.addData('advanceScreenPress4.keys',advanceScreenPress4.keys)
    if advanceScreenPress4.keys != None:  # we had a response
        thisExp.addData('advanceScreenPress4.rt', advanceScreenPress4.rt)
        thisExp.addData('advanceScreenPress4.duration', advanceScreenPress4.duration)
    thisExp.nextEntry()
    # the Routine "summaryDirections" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    runs = data.TrialHandler2(
        name='runs',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions/Runs.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(runs)  # add the loop to the experiment
    thisRun = runs.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
    if thisRun != None:
        for paramName in thisRun:
            globals()[paramName] = thisRun[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRun in runs:
        runs.status = STARTED
        if hasattr(thisRun, 'status'):
            thisRun.status = STARTED
        currentLoop = runs
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
        if thisRun != None:
            for paramName in thisRun:
                globals()[paramName] = thisRun[paramName]
        
        # --- Prepare to start Routine "getReady" ---
        # create an object to store info about Routine getReady
        getReady = data.Routine(
            name='getReady',
            components=[getReadyText, advanceToTrigger],
        )
        getReady.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for advanceToTrigger
        advanceToTrigger.keys = []
        advanceToTrigger.rt = []
        _advanceToTrigger_allKeys = []
        # Run 'Begin Routine' code from setRunFiles
        print(Runs)
        if expInfo["startFromRun"] == "1":
            if Runs == 'Run1':
                trialOrder = milConditionsFile1
            if Runs == 'Run2':
                trialOrder = milConditionsFile2
        else:
            if Runs == 'Run1':
                trialOrder = milConditionsFile1 #which has been set to 2 earlier
        
        
        
        # store start times for getReady
        getReady.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        getReady.tStart = globalClock.getTime(format='float')
        getReady.status = STARTED
        thisExp.addData('getReady.started', getReady.tStart)
        getReady.maxDuration = None
        # keep track of which components have finished
        getReadyComponents = getReady.components
        for thisComponent in getReady.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "getReady" ---
        getReady.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRun, 'status') and thisRun.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *getReadyText* updates
            
            # if getReadyText is starting this frame...
            if getReadyText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                getReadyText.frameNStart = frameN  # exact frame index
                getReadyText.tStart = t  # local t and not account for scr refresh
                getReadyText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(getReadyText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'getReadyText.started')
                # update status
                getReadyText.status = STARTED
                getReadyText.setAutoDraw(True)
            
            # if getReadyText is active this frame...
            if getReadyText.status == STARTED:
                # update params
                pass
            
            # *advanceToTrigger* updates
            waitOnFlip = False
            
            # if advanceToTrigger is starting this frame...
            if advanceToTrigger.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                advanceToTrigger.frameNStart = frameN  # exact frame index
                advanceToTrigger.tStart = t  # local t and not account for scr refresh
                advanceToTrigger.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(advanceToTrigger, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'advanceToTrigger.started')
                # update status
                advanceToTrigger.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(advanceToTrigger.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(advanceToTrigger.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if advanceToTrigger.status == STARTED and not waitOnFlip:
                theseKeys = advanceToTrigger.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _advanceToTrigger_allKeys.extend(theseKeys)
                if len(_advanceToTrigger_allKeys):
                    advanceToTrigger.keys = _advanceToTrigger_allKeys[-1].name  # just the last key pressed
                    advanceToTrigger.rt = _advanceToTrigger_allKeys[-1].rt
                    advanceToTrigger.duration = _advanceToTrigger_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=getReady,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                getReady.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in getReady.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "getReady" ---
        for thisComponent in getReady.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for getReady
        getReady.tStop = globalClock.getTime(format='float')
        getReady.tStopRefresh = tThisFlipGlobal
        thisExp.addData('getReady.stopped', getReady.tStop)
        # check responses
        if advanceToTrigger.keys in ['', [], None]:  # No response was made
            advanceToTrigger.keys = None
        runs.addData('advanceToTrigger.keys',advanceToTrigger.keys)
        if advanceToTrigger.keys != None:  # we had a response
            runs.addData('advanceToTrigger.rt', advanceToTrigger.rt)
            runs.addData('advanceToTrigger.duration', advanceToTrigger.duration)
        # the Routine "getReady" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "waitForScanner" ---
        # create an object to store info about Routine waitForScanner
        waitForScanner = data.Routine(
            name='waitForScanner',
            components=[scannerTriggerKey, sumTemplate, hg1, hg2, lg1, lg2, hl1, hl2, ll1, ll2],
        )
        waitForScanner.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from waitScannerCode
        #skip scanner trigger if 
        if(expInfo["mode"] != "scan"):
            continueRoutine = False
        
        '''
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
        
        '''
        # create starting attributes for scannerTriggerKey
        scannerTriggerKey.keys = []
        scannerTriggerKey.rt = []
        _scannerTriggerKey_allKeys = []
        # Run 'Begin Routine' code from setImg
        hg1Img = 'stimuli_sets/' + cuePairs['highgain']['optimalChoice']
        hg2Img = 'stimuli_sets/' + cuePairs['highgain']['suboptimalChoice']
        
        lg1Img = 'stimuli_sets/' + cuePairs['lowgain']['suboptimalChoice']
        lg2Img = 'stimuli_sets/' + cuePairs['lowgain']['optimalChoice']
        
        hl1Img = 'stimuli_sets/' + cuePairs['highloss']['suboptimalChoice']
        hl2Img = 'stimuli_sets/' + cuePairs['highloss']['optimalChoice']
        
        ll1Img = 'stimuli_sets/' + cuePairs['lowloss']['optimalChoice']
        ll2Img = 'stimuli_sets/' + cuePairs['lowloss']['suboptimalChoice']
        hg1.setImage(hg1Img)
        hg2.setImage(hg2Img)
        lg1.setImage(lg1Img)
        lg2.setImage(lg2Img)
        hl1.setImage(hl1Img)
        hl2.setImage(hl2Img)
        ll1.setImage(ll1Img)
        ll2.setImage(ll2Img)
        # store start times for waitForScanner
        waitForScanner.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        waitForScanner.tStart = globalClock.getTime(format='float')
        waitForScanner.status = STARTED
        thisExp.addData('waitForScanner.started', waitForScanner.tStart)
        waitForScanner.maxDuration = None
        # keep track of which components have finished
        waitForScannerComponents = waitForScanner.components
        for thisComponent in waitForScanner.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "waitForScanner" ---
        waitForScanner.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRun, 'status') and thisRun.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *scannerTriggerKey* updates
            waitOnFlip = False
            
            # if scannerTriggerKey is starting this frame...
            if scannerTriggerKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                scannerTriggerKey.frameNStart = frameN  # exact frame index
                scannerTriggerKey.tStart = t  # local t and not account for scr refresh
                scannerTriggerKey.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(scannerTriggerKey, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'scannerTriggerKey.started')
                # update status
                scannerTriggerKey.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(scannerTriggerKey.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(scannerTriggerKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if scannerTriggerKey.status == STARTED and not waitOnFlip:
                theseKeys = scannerTriggerKey.getKeys(keyList=['t'], ignoreKeys=["escape"], waitRelease=False)
                _scannerTriggerKey_allKeys.extend(theseKeys)
                if len(_scannerTriggerKey_allKeys):
                    scannerTriggerKey.keys = _scannerTriggerKey_allKeys[-1].name  # just the last key pressed
                    scannerTriggerKey.rt = _scannerTriggerKey_allKeys[-1].rt
                    scannerTriggerKey.duration = _scannerTriggerKey_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *sumTemplate* updates
            
            # if sumTemplate is starting this frame...
            if sumTemplate.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sumTemplate.frameNStart = frameN  # exact frame index
                sumTemplate.tStart = t  # local t and not account for scr refresh
                sumTemplate.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sumTemplate, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sumTemplate.started')
                # update status
                sumTemplate.status = STARTED
                sumTemplate.setAutoDraw(True)
            
            # if sumTemplate is active this frame...
            if sumTemplate.status == STARTED:
                # update params
                pass
            
            # *hg1* updates
            
            # if hg1 is starting this frame...
            if hg1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                hg1.frameNStart = frameN  # exact frame index
                hg1.tStart = t  # local t and not account for scr refresh
                hg1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(hg1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'hg1.started')
                # update status
                hg1.status = STARTED
                hg1.setAutoDraw(True)
            
            # if hg1 is active this frame...
            if hg1.status == STARTED:
                # update params
                pass
            
            # *hg2* updates
            
            # if hg2 is starting this frame...
            if hg2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                hg2.frameNStart = frameN  # exact frame index
                hg2.tStart = t  # local t and not account for scr refresh
                hg2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(hg2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'hg2.started')
                # update status
                hg2.status = STARTED
                hg2.setAutoDraw(True)
            
            # if hg2 is active this frame...
            if hg2.status == STARTED:
                # update params
                pass
            
            # *lg1* updates
            
            # if lg1 is starting this frame...
            if lg1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lg1.frameNStart = frameN  # exact frame index
                lg1.tStart = t  # local t and not account for scr refresh
                lg1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lg1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lg1.started')
                # update status
                lg1.status = STARTED
                lg1.setAutoDraw(True)
            
            # if lg1 is active this frame...
            if lg1.status == STARTED:
                # update params
                pass
            
            # *lg2* updates
            
            # if lg2 is starting this frame...
            if lg2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lg2.frameNStart = frameN  # exact frame index
                lg2.tStart = t  # local t and not account for scr refresh
                lg2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lg2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lg2.started')
                # update status
                lg2.status = STARTED
                lg2.setAutoDraw(True)
            
            # if lg2 is active this frame...
            if lg2.status == STARTED:
                # update params
                pass
            
            # *hl1* updates
            
            # if hl1 is starting this frame...
            if hl1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                hl1.frameNStart = frameN  # exact frame index
                hl1.tStart = t  # local t and not account for scr refresh
                hl1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(hl1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'hl1.started')
                # update status
                hl1.status = STARTED
                hl1.setAutoDraw(True)
            
            # if hl1 is active this frame...
            if hl1.status == STARTED:
                # update params
                pass
            
            # *hl2* updates
            
            # if hl2 is starting this frame...
            if hl2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                hl2.frameNStart = frameN  # exact frame index
                hl2.tStart = t  # local t and not account for scr refresh
                hl2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(hl2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'hl2.started')
                # update status
                hl2.status = STARTED
                hl2.setAutoDraw(True)
            
            # if hl2 is active this frame...
            if hl2.status == STARTED:
                # update params
                pass
            
            # *ll1* updates
            
            # if ll1 is starting this frame...
            if ll1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ll1.frameNStart = frameN  # exact frame index
                ll1.tStart = t  # local t and not account for scr refresh
                ll1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ll1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'll1.started')
                # update status
                ll1.status = STARTED
                ll1.setAutoDraw(True)
            
            # if ll1 is active this frame...
            if ll1.status == STARTED:
                # update params
                pass
            
            # *ll2* updates
            
            # if ll2 is starting this frame...
            if ll2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ll2.frameNStart = frameN  # exact frame index
                ll2.tStart = t  # local t and not account for scr refresh
                ll2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ll2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'll2.started')
                # update status
                ll2.status = STARTED
                ll2.setAutoDraw(True)
            
            # if ll2 is active this frame...
            if ll2.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=waitForScanner,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                waitForScanner.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in waitForScanner.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "waitForScanner" ---
        for thisComponent in waitForScanner.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for waitForScanner
        waitForScanner.tStop = globalClock.getTime(format='float')
        waitForScanner.tStopRefresh = tThisFlipGlobal
        thisExp.addData('waitForScanner.stopped', waitForScanner.tStop)
        # Run 'End Routine' code from waitScannerCode
        #routineTimer.reset()
        fmriClock.reset()
        # check responses
        if scannerTriggerKey.keys in ['', [], None]:  # No response was made
            scannerTriggerKey.keys = None
        runs.addData('scannerTriggerKey.keys',scannerTriggerKey.keys)
        if scannerTriggerKey.keys != None:  # we had a response
            runs.addData('scannerTriggerKey.rt', scannerTriggerKey.rt)
            runs.addData('scannerTriggerKey.duration', scannerTriggerKey.duration)
        # Run 'End Routine' code from frameOnCode
        frame.setAutoDraw(True)
        # the Routine "waitForScanner" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(trialOrder), 
            seed=None, 
        )
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            trials.status = STARTED
            if hasattr(thisTrial, 'status'):
                thisTrial.status = STARTED
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "cue" ---
            # create an object to store info about Routine cue
            cue = data.Routine(
                name='cue',
                components=[startFixStatic, leftCue, rightCue, cueResp, staticISI],
            )
            cue.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for cueResp
            cueResp.keys = []
            cueResp.rt = []
            _cueResp_allKeys = []
            # allowedKeys looks like a variable, so make sure it exists locally
            if 'allowedKeys' in globals():
                allowedKeys = globals()['allowedKeys']
            # Run 'Begin Routine' code from setCues
            logging.info("Iterating through thisTrial")
            for value in trials.trialList:
                logging.info(value)
            
            # update component parameters for each repeat
            optimalImg = cuePairs[condition]['optimalChoice']
            suboptimalImg = cuePairs[condition]['suboptimalChoice']
            currentLoop.addData('optimalImg', optimalImg)
            currentLoop.addData('suboptimalImg', suboptimalImg)
            
            
            # Assumes button box responses are '1' for index and '2' for middle finger.
            if np.random.uniform() > 0.5:
                cuePositions = {responseKeys['left']:optimalImg, responseKeys['right']:suboptimalImg}
            
            else:
                cuePositions =  {responseKeys['left']:suboptimalImg, responseKeys['right']:optimalImg}
                
            sidePositions = dict([(v, k) for (k, v) in cuePositions.items()])
            
            optimalSide='left' if sidePositions[optimalImg]==responseKeys['left'] else 'right'
            suboptimalSide='left' if sidePositions[suboptimalImg]==responseKeys['left'] else 'right'
            
            currentLoop.addData('optimalSide', optimalSide)
            currentLoop.addData('suboptimalSide', suboptimalSide)
            currentLoop.addData('optimalResponse',sidePositions[optimalImg])
            currentLoop.addData('suboptimalResponse',sidePositions[suboptimalImg])
            leftImage = 'stimuli_sets/'+cuePositions[responseKeys['left']]
            currentLoop.addData('leftImage', leftImage)
            rightImage = 'stimuli_sets/'+cuePositions[responseKeys['right']]
            currentLoop.addData('rightImage', rightImage)
            
            # Set the "correct" response by choosing the one that will give the better outcome.
            # That's a higher probability of winning in gain conditions and a lower probability 
            # of losing in loss conditions. Used for analysis, not for choosing outcomes.
            correct = sidePositions[optimalImg]
            # Run 'Begin Routine' code from frameLabels
            if condition == "lowgain":
                corrvalue=0.20
                incorrvalue=0
                topLabel="+$%.2f"%(abs(corrvalue))
                bottomLabel="+$%.2f"%(abs(incorrvalue))
            
            if condition == "highgain":
                corrvalue=1.00
                incorrvalue=0
                topLabel="+$%.2f"%(abs(corrvalue))
                bottomLabel="+$%.2f"%(abs(incorrvalue))
            
            if condition == "lowloss":
                corrvalue=0
                incorrvalue=-0.20
                topLabel="-$%.2f"%(abs(corrvalue))
                bottomLabel="-$%.2f"%(abs(incorrvalue))
            
            if condition == "highloss":
                corrvalue=0
                incorrvalue=-1.00
                topLabel="-$%.2f"%(abs(corrvalue))
                bottomLabel="-$%.2f"%(abs(incorrvalue))
                
            topFrameText.setText(topLabel)
            topFrameText.status = NOT_STARTED
            bottomFrameText.setText(bottomLabel)
            bottomFrameText.status = NOT_STARTED
            # Run 'Begin Routine' code from setImageTiming
            cueTimeAdded = False
            responseTimeAdded = False
            feedbackTimeAdded = False
            currentLoop.addData('startFixOnTime', fmriClock.getTime())
            # Run 'Begin Routine' code from manageTrialsLoop
            trialsClock.reset()
            # Run 'Begin Routine' code from selectionIndicatorCode
            # Initialize cue end time for cues and selection indicator.
            # If no response is made in this time, the trial ends.
            # Otherwise, cueEndTime is updated to response
            # time plus the selection duration.
            cueEndTime = 3
            selectionIndicator.status = NOT_STARTED
            # store start times for cue
            cue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            cue.tStart = globalClock.getTime(format='float')
            cue.status = STARTED
            thisExp.addData('cue.started', cue.tStart)
            cue.maxDuration = None
            # keep track of which components have finished
            cueComponents = cue.components
            for thisComponent in cue.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "cue" ---
            cue.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 3.0:
                # if trial has changed, end Routine now
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *startFixStatic* updates
                
                # if startFixStatic is starting this frame...
                if startFixStatic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    startFixStatic.frameNStart = frameN  # exact frame index
                    startFixStatic.tStart = t  # local t and not account for scr refresh
                    startFixStatic.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(startFixStatic, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'startFixStatic.started')
                    # update status
                    startFixStatic.status = STARTED
                    startFixStatic.setAutoDraw(True)
                
                # if startFixStatic is active this frame...
                if startFixStatic.status == STARTED:
                    # update params
                    pass
                
                # if startFixStatic is stopping this frame...
                if startFixStatic.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > startFixStatic.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        startFixStatic.tStop = t  # not accounting for scr refresh
                        startFixStatic.tStopRefresh = tThisFlipGlobal  # on global time
                        startFixStatic.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'startFixStatic.stopped')
                        # update status
                        startFixStatic.status = FINISHED
                        startFixStatic.setAutoDraw(False)
                
                # *leftCue* updates
                
                # if leftCue is starting this frame...
                if leftCue.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    leftCue.frameNStart = frameN  # exact frame index
                    leftCue.tStart = t  # local t and not account for scr refresh
                    leftCue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(leftCue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'leftCue.started')
                    # update status
                    leftCue.status = STARTED
                    leftCue.setAutoDraw(True)
                
                # if leftCue is active this frame...
                if leftCue.status == STARTED:
                    # update params
                    pass
                
                # if leftCue is stopping this frame...
                if leftCue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > leftCue.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        leftCue.tStop = t  # not accounting for scr refresh
                        leftCue.tStopRefresh = tThisFlipGlobal  # on global time
                        leftCue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'leftCue.stopped')
                        # update status
                        leftCue.status = FINISHED
                        leftCue.setAutoDraw(False)
                
                # *rightCue* updates
                
                # if rightCue is starting this frame...
                if rightCue.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    rightCue.frameNStart = frameN  # exact frame index
                    rightCue.tStart = t  # local t and not account for scr refresh
                    rightCue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(rightCue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rightCue.started')
                    # update status
                    rightCue.status = STARTED
                    rightCue.setAutoDraw(True)
                
                # if rightCue is active this frame...
                if rightCue.status == STARTED:
                    # update params
                    pass
                
                # if rightCue is stopping this frame...
                if rightCue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > rightCue.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        rightCue.tStop = t  # not accounting for scr refresh
                        rightCue.tStopRefresh = tThisFlipGlobal  # on global time
                        rightCue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'rightCue.stopped')
                        # update status
                        rightCue.status = FINISHED
                        rightCue.setAutoDraw(False)
                
                # *cueResp* updates
                waitOnFlip = False
                
                # if cueResp is starting this frame...
                if cueResp.status == NOT_STARTED and tThisFlip >= .5-frameTolerance:
                    # keep track of start time/frame for later
                    cueResp.frameNStart = frameN  # exact frame index
                    cueResp.tStart = t  # local t and not account for scr refresh
                    cueResp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cueResp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cueResp.started')
                    # update status
                    cueResp.status = STARTED
                    # allowed keys looks like a variable named `allowedKeys`
                    if not type(allowedKeys) in [list, tuple, np.ndarray]:
                        if not isinstance(allowedKeys, str):
                            allowedKeys = str(allowedKeys)
                        elif not ',' in allowedKeys:
                            allowedKeys = (allowedKeys,)
                        else:
                            allowedKeys = eval(allowedKeys)
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(cueResp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(cueResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if cueResp is stopping this frame...
                if cueResp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cueResp.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        cueResp.tStop = t  # not accounting for scr refresh
                        cueResp.tStopRefresh = tThisFlipGlobal  # on global time
                        cueResp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cueResp.stopped')
                        # update status
                        cueResp.status = FINISHED
                        cueResp.status = FINISHED
                if cueResp.status == STARTED and not waitOnFlip:
                    theseKeys = cueResp.getKeys(keyList=list(allowedKeys), ignoreKeys=["escape"], waitRelease=False)
                    _cueResp_allKeys.extend(theseKeys)
                    if len(_cueResp_allKeys):
                        cueResp.keys = _cueResp_allKeys[0].name  # just the first key pressed
                        cueResp.rt = _cueResp_allKeys[0].rt
                        cueResp.duration = _cueResp_allKeys[0].duration
                        # was this correct?
                        if (cueResp.keys == str(correct)) or (cueResp.keys == correct):
                            cueResp.corr = 1
                        else:
                            cueResp.corr = 0
                # Run 'Each Frame' code from frameLabels
                if t > .5 and topFrameText.status == NOT_STARTED:
                    topFrameText.setAutoDraw(True)
                    bottomFrameText.setAutoDraw(True)
                    topFrameText.status = STARTED
                # Run 'Each Frame' code from setImageTiming
                if not cueTimeAdded: 
                    if leftCue.autoDraw==True:
                        currentLoop.addData('cueOnTime', fmriClock.getTime())
                        cueTimeAdded = True
                        
                #if not chooseTimeAdded and chooseText.status == STARTED:
                #    currentLoop.addData('chooseTime', fmriClock.getTime())
                #    chooseTimeAdded = True
                if not responseTimeAdded and len(cueResp.keys):
                    currentLoop.addData('responseTime', fmriClock.getTime())
                    responseTimeAdded = True
                # Run 'Each Frame' code from selectionIndicatorCode
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
                # *staticISI* period
                
                # if staticISI is starting this frame...
                if staticISI.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    staticISI.frameNStart = frameN  # exact frame index
                    staticISI.tStart = t  # local t and not account for scr refresh
                    staticISI.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(staticISI, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('staticISI.started', t)
                    # update status
                    staticISI.status = STARTED
                    # start the static period
                    staticISI.start(0.5)
                elif staticISI.status == STARTED:  # one frame should pass before updating params and completing
                    # Updating other components during *staticISI*
                    leftCue.setImage(leftImage)
                    rightCue.setImage(rightImage)
                    # Component updates done
                    # finish the static period and store its duration
                    staticISI.complete()
                    staticISI.tStop = staticISI.tStart + staticISI.getDuration()
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=cue,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    cue.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in cue.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "cue" ---
            for thisComponent in cue.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for cue
            cue.tStop = globalClock.getTime(format='float')
            cue.tStopRefresh = tThisFlipGlobal
            thisExp.addData('cue.stopped', cue.tStop)
            # check responses
            if cueResp.keys in ['', [], None]:  # No response was made
                cueResp.keys = None
                # was no response the correct answer?!
                if str(correct).lower() == 'none':
                   cueResp.corr = 1;  # correct non-response
                else:
                   cueResp.corr = 0;  # failed to respond (incorrectly)
            # store data for trials (TrialHandler)
            trials.addData('cueResp.keys',cueResp.keys)
            trials.addData('cueResp.corr', cueResp.corr)
            if cueResp.keys != None:  # we had a response
                trials.addData('cueResp.rt', cueResp.rt)
                trials.addData('cueResp.duration', cueResp.duration)
            # Run 'End Routine' code from setCues
            if cueResp.keys == responseKeys['left']:
                currentLoop.addData('optedSide', 'left')
                currentLoop.addData('optedImg', leftImage)
                
            elif cueResp.keys == responseKeys['right']:
                currentLoop.addData('optedSide', 'right')
                currentLoop.addData('optedImg', rightImage)
                
            else: 
                currentLoop.addData('optedSide', 'NA')
                currentLoop.addData('optedImg', 'NA')
            # Run 'End Routine' code from frameLabels
            topFrameText.status = NOT_STARTED
            # Run 'End Routine' code from setImageTiming
            currentLoop.addData('cueOffTime', fmriClock.getTime())
            
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if cue.maxDurationReached:
                routineTimer.addTime(-cue.maxDuration)
            elif cue.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)
            
            # --- Prepare to start Routine "outcomeDelayRoutine" ---
            # create an object to store info about Routine outcomeDelayRoutine
            outcomeDelayRoutine = data.Routine(
                name='outcomeDelayRoutine',
                components=[outcomeDelayFix],
            )
            outcomeDelayRoutine.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for outcomeDelayRoutine
            outcomeDelayRoutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            outcomeDelayRoutine.tStart = globalClock.getTime(format='float')
            outcomeDelayRoutine.status = STARTED
            thisExp.addData('outcomeDelayRoutine.started', outcomeDelayRoutine.tStart)
            outcomeDelayRoutine.maxDuration = None
            # keep track of which components have finished
            outcomeDelayRoutineComponents = outcomeDelayRoutine.components
            for thisComponent in outcomeDelayRoutine.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "outcomeDelayRoutine" ---
            outcomeDelayRoutine.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *outcomeDelayFix* updates
                
                # if outcomeDelayFix is starting this frame...
                if outcomeDelayFix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    outcomeDelayFix.frameNStart = frameN  # exact frame index
                    outcomeDelayFix.tStart = t  # local t and not account for scr refresh
                    outcomeDelayFix.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(outcomeDelayFix, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'outcomeDelayFix.started')
                    # update status
                    outcomeDelayFix.status = STARTED
                    outcomeDelayFix.setAutoDraw(True)
                
                # if outcomeDelayFix is active this frame...
                if outcomeDelayFix.status == STARTED:
                    # update params
                    pass
                
                # if outcomeDelayFix is stopping this frame...
                if outcomeDelayFix.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > outcomeDelayFix.tStartRefresh + outcomeDelay-frameTolerance:
                        # keep track of stop time/frame for later
                        outcomeDelayFix.tStop = t  # not accounting for scr refresh
                        outcomeDelayFix.tStopRefresh = tThisFlipGlobal  # on global time
                        outcomeDelayFix.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'outcomeDelayFix.stopped')
                        # update status
                        outcomeDelayFix.status = FINISHED
                        outcomeDelayFix.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=outcomeDelayRoutine,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    outcomeDelayRoutine.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in outcomeDelayRoutine.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "outcomeDelayRoutine" ---
            for thisComponent in outcomeDelayRoutine.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for outcomeDelayRoutine
            outcomeDelayRoutine.tStop = globalClock.getTime(format='float')
            outcomeDelayRoutine.tStopRefresh = tThisFlipGlobal
            thisExp.addData('outcomeDelayRoutine.stopped', outcomeDelayRoutine.tStop)
            # the Routine "outcomeDelayRoutine" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "outcomeRoutine" ---
            # create an object to store info about Routine outcomeRoutine
            outcomeRoutine = data.Routine(
                name='outcomeRoutine',
                components=[outcomeText],
            )
            outcomeRoutine.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from rightChoice
            if cueResp.keys==sidePositions[optimalImg]:
                Choice="optimal"
            elif cueResp.keys==sidePositions[suboptimalImg]:
                Choice="suboptimal"
            else: 
                Choice="NA"
                
            Accuracy=cueResp.corr
            
            currentLoop.addData('optedFor', Choice)
            currentLoop.addData('accuracy', Accuracy)
            # Run 'Begin Routine' code from defineOutcomes
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
                    outcome_msg = "%s$%.2f\nCorrect"%(trueOperator, abs(change))
                    outcome_image = None
                    Feedback=1
            
                else: 
                    # If outcome was False, the trial doesn't change anything.
                    # Still display the operator for trial condition
                    change = incorrvalue
                    outcome_msg = "%s$%.2f\nIncorrect" % (falseOperator,abs(change))
                    outcome_image = None
                    Feedback=0
            else: # give incorrect outcome for missed trials
                change = incorrvalue
                currentLoop.addData('outcome', None)
                outcome_msg = "%s$%.2f\nMissed" % (falseOperator,abs(change))
                outcome_image = None
                Feedback="NA"
            
            
            
            # Add a positive or negative change
            bank += change
            
            currentLoop.addData('Feedback', Feedback)
            currentLoop.addData('gainLossAmount', change)
            currentLoop.addData('runningBankTotal', bank)
            currentLoop.addData('feedbackTime', fmriClock.getTime())
            
            outcomeText.setText(outcome_msg)
            # store start times for outcomeRoutine
            outcomeRoutine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            outcomeRoutine.tStart = globalClock.getTime(format='float')
            outcomeRoutine.status = STARTED
            thisExp.addData('outcomeRoutine.started', outcomeRoutine.tStart)
            outcomeRoutine.maxDuration = None
            # keep track of which components have finished
            outcomeRoutineComponents = outcomeRoutine.components
            for thisComponent in outcomeRoutine.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "outcomeRoutine" ---
            outcomeRoutine.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # if trial has changed, end Routine now
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *outcomeText* updates
                
                # if outcomeText is starting this frame...
                if outcomeText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    outcomeText.frameNStart = frameN  # exact frame index
                    outcomeText.tStart = t  # local t and not account for scr refresh
                    outcomeText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(outcomeText, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'outcomeText.started')
                    # update status
                    outcomeText.status = STARTED
                    outcomeText.setAutoDraw(True)
                
                # if outcomeText is active this frame...
                if outcomeText.status == STARTED:
                    # update params
                    pass
                
                # if outcomeText is stopping this frame...
                if outcomeText.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > outcomeText.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        outcomeText.tStop = t  # not accounting for scr refresh
                        outcomeText.tStopRefresh = tThisFlipGlobal  # on global time
                        outcomeText.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'outcomeText.stopped')
                        # update status
                        outcomeText.status = FINISHED
                        outcomeText.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=outcomeRoutine,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    outcomeRoutine.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in outcomeRoutine.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "outcomeRoutine" ---
            for thisComponent in outcomeRoutine.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for outcomeRoutine
            outcomeRoutine.tStop = globalClock.getTime(format='float')
            outcomeRoutine.tStopRefresh = tThisFlipGlobal
            thisExp.addData('outcomeRoutine.stopped', outcomeRoutine.tStop)
            # Run 'End Routine' code from defineOutcomes
            expInfo['bank'] = bank
            currentLoop.addData('feedbackOffTime', fmriClock.getTime())
            # Run 'End Routine' code from frameLabelsOff
            topFrameText.setAutoDraw(False)
            bottomFrameText.setAutoDraw(False)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if outcomeRoutine.maxDurationReached:
                routineTimer.addTime(-outcomeRoutine.maxDuration)
            elif outcomeRoutine.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "fixation" ---
            # create an object to store info about Routine fixation
            fixation = data.Routine(
                name='fixation',
                components=[fixationCrosshair],
            )
            fixation.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from loopDurationCode
            currentLoop.addData('fixationTime', fmriClock.getTime())
            # store start times for fixation
            fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            fixation.tStart = globalClock.getTime(format='float')
            fixation.status = STARTED
            thisExp.addData('fixation.started', fixation.tStart)
            fixation.maxDuration = None
            # keep track of which components have finished
            fixationComponents = fixation.components
            for thisComponent in fixation.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation" ---
            fixation.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixationCrosshair* updates
                
                # if fixationCrosshair is starting this frame...
                if fixationCrosshair.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixationCrosshair.frameNStart = frameN  # exact frame index
                    fixationCrosshair.tStart = t  # local t and not account for scr refresh
                    fixationCrosshair.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixationCrosshair, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixationCrosshair.started')
                    # update status
                    fixationCrosshair.status = STARTED
                    fixationCrosshair.setAutoDraw(True)
                
                # if fixationCrosshair is active this frame...
                if fixationCrosshair.status == STARTED:
                    # update params
                    pass
                
                # if fixationCrosshair is stopping this frame...
                if fixationCrosshair.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixationCrosshair.tStartRefresh + fixationDuration-frameTolerance:
                        # keep track of stop time/frame for later
                        fixationCrosshair.tStop = t  # not accounting for scr refresh
                        fixationCrosshair.tStopRefresh = tThisFlipGlobal  # on global time
                        fixationCrosshair.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixationCrosshair.stopped')
                        # update status
                        fixationCrosshair.status = FINISHED
                        fixationCrosshair.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=fixation,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    fixation.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixation.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation" ---
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for fixation
            fixation.tStop = globalClock.getTime(format='float')
            fixation.tStopRefresh = tThisFlipGlobal
            thisExp.addData('fixation.stopped', fixation.tStop)
            # Run 'End Routine' code from loopDurationCode
            logging.exp("Measured Fixation Dureation: %f" % t)
            logging.exp("Trials Clock: %f" % trialsClock.getTime())
            # the Routine "fixation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisTrial as finished
            if hasattr(thisTrial, 'status'):
                thisTrial.status = FINISHED
            # if awaiting a pause, pause now
            if trials.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                trials.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials'
        trials.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if trials.trialList in ([], [None], None):
            params = []
        else:
            params = trials.trialList[0].keys()
        # save data for this loop
        trials.saveAsText(filename + '_trials.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "breakScreen" ---
        # create an object to store info about Routine breakScreen
        breakScreen = data.Routine(
            name='breakScreen',
            components=[breakText, advanceScreenPress_2],
        )
        breakScreen.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for advanceScreenPress_2
        advanceScreenPress_2.keys = []
        advanceScreenPress_2.rt = []
        _advanceScreenPress_2_allKeys = []
        # Run 'Begin Routine' code from endRunTime
        currentLoop.addData('runOffTime', fmriClock.getTime())
        if(expInfo["startFromRun"] == "2"):
            print("trying to end loop")
            trials.finished = True
            runs.finished = True
            continueRoutine = False  #add this line to end the current routine early
        else:
            if runs.thisN == 1:
                continueRoutine = False 
        #if expInfo["startFromRun"] == "2":
        #   print("trying to end this damn round")
        #   #continueRoutine = False
        #   runs.finished == True
        # Run 'Begin Routine' code from frameOffCode
        frame.setAutoDraw(False)
        # store start times for breakScreen
        breakScreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        breakScreen.tStart = globalClock.getTime(format='float')
        breakScreen.status = STARTED
        thisExp.addData('breakScreen.started', breakScreen.tStart)
        breakScreen.maxDuration = None
        # keep track of which components have finished
        breakScreenComponents = breakScreen.components
        for thisComponent in breakScreen.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "breakScreen" ---
        breakScreen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRun, 'status') and thisRun.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *breakText* updates
            
            # if breakText is starting this frame...
            if breakText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                breakText.frameNStart = frameN  # exact frame index
                breakText.tStart = t  # local t and not account for scr refresh
                breakText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(breakText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'breakText.started')
                # update status
                breakText.status = STARTED
                breakText.setAutoDraw(True)
            
            # if breakText is active this frame...
            if breakText.status == STARTED:
                # update params
                pass
            
            # *advanceScreenPress_2* updates
            waitOnFlip = False
            
            # if advanceScreenPress_2 is starting this frame...
            if advanceScreenPress_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                advanceScreenPress_2.frameNStart = frameN  # exact frame index
                advanceScreenPress_2.tStart = t  # local t and not account for scr refresh
                advanceScreenPress_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(advanceScreenPress_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'advanceScreenPress_2.started')
                # update status
                advanceScreenPress_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(advanceScreenPress_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(advanceScreenPress_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if advanceScreenPress_2.status == STARTED and not waitOnFlip:
                theseKeys = advanceScreenPress_2.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _advanceScreenPress_2_allKeys.extend(theseKeys)
                if len(_advanceScreenPress_2_allKeys):
                    advanceScreenPress_2.keys = _advanceScreenPress_2_allKeys[-1].name  # just the last key pressed
                    advanceScreenPress_2.rt = _advanceScreenPress_2_allKeys[-1].rt
                    advanceScreenPress_2.duration = _advanceScreenPress_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=breakScreen,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                breakScreen.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in breakScreen.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "breakScreen" ---
        for thisComponent in breakScreen.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for breakScreen
        breakScreen.tStop = globalClock.getTime(format='float')
        breakScreen.tStopRefresh = tThisFlipGlobal
        thisExp.addData('breakScreen.stopped', breakScreen.tStop)
        # check responses
        if advanceScreenPress_2.keys in ['', [], None]:  # No response was made
            advanceScreenPress_2.keys = None
        runs.addData('advanceScreenPress_2.keys',advanceScreenPress_2.keys)
        if advanceScreenPress_2.keys != None:  # we had a response
            runs.addData('advanceScreenPress_2.rt', advanceScreenPress_2.rt)
            runs.addData('advanceScreenPress_2.duration', advanceScreenPress_2.duration)
        # the Routine "breakScreen" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisRun as finished
        if hasattr(thisRun, 'status'):
            thisRun.status = FINISHED
        # if awaiting a pause, pause now
        if runs.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            runs.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'runs'
    runs.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if runs.trialList in ([], [None], None):
        params = []
    else:
        params = runs.trialList[0].keys()
    # save data for this loop
    runs.saveAsText(filename + '_runs.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "check6" ---
    # create an object to store info about Routine check6
    check6 = data.Routine(
        name='check6',
        components=[check6Text, check6ImageLeft, check6ImageRight, check6Press],
    )
    check6.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    check6ImageLeft.setImage(hg1Img)
    check6ImageRight.setImage(hg2Img)
    # Run 'Begin Routine' code from check6Select
    # Initialize cue end time for cues and selection indicator.
    # If no response is made in this time, the trial ends.
    # Otherwise, cueEndTime is updated to response
    # time plus the selection duration.
    
    selectionIndicatorPrac.status = NOT_STARTED
    t_started = False
    # create starting attributes for check6Press
    check6Press.keys = []
    check6Press.rt = []
    _check6Press_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'allowedKeys' in globals():
        allowedKeys = globals()['allowedKeys']
    # store start times for check6
    check6.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    check6.tStart = globalClock.getTime(format='float')
    check6.status = STARTED
    thisExp.addData('check6.started', check6.tStart)
    check6.maxDuration = None
    # keep track of which components have finished
    check6Components = check6.components
    for thisComponent in check6.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "check6" ---
    check6.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *check6Text* updates
        
        # if check6Text is starting this frame...
        if check6Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check6Text.frameNStart = frameN  # exact frame index
            check6Text.tStart = t  # local t and not account for scr refresh
            check6Text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check6Text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check6Text.started')
            # update status
            check6Text.status = STARTED
            check6Text.setAutoDraw(True)
        
        # if check6Text is active this frame...
        if check6Text.status == STARTED:
            # update params
            pass
        
        # *check6ImageLeft* updates
        
        # if check6ImageLeft is starting this frame...
        if check6ImageLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check6ImageLeft.frameNStart = frameN  # exact frame index
            check6ImageLeft.tStart = t  # local t and not account for scr refresh
            check6ImageLeft.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check6ImageLeft, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check6ImageLeft.started')
            # update status
            check6ImageLeft.status = STARTED
            check6ImageLeft.setAutoDraw(True)
        
        # if check6ImageLeft is active this frame...
        if check6ImageLeft.status == STARTED:
            # update params
            pass
        
        # *check6ImageRight* updates
        
        # if check6ImageRight is starting this frame...
        if check6ImageRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check6ImageRight.frameNStart = frameN  # exact frame index
            check6ImageRight.tStart = t  # local t and not account for scr refresh
            check6ImageRight.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check6ImageRight, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check6ImageRight.started')
            # update status
            check6ImageRight.status = STARTED
            check6ImageRight.setAutoDraw(True)
        
        # if check6ImageRight is active this frame...
        if check6ImageRight.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from check6Select
        # If the selection has been drawn for the selection duration,
        # or the trial should end (cutting off the selection duration), end the routine
        if t_started:
            if t > t_started + 5:
                continueRoutine = False
        
        # Start Drawing Selection Indicator if Selection has been made
        if check6Press.keys and selectionIndicatorPrac.status == NOT_STARTED:
            #cueComponents.append(selectionIndicatorPrac)
            selectionIndicatorPrac.status = STARTED
            selectionIndicatorPrac.setPos(selectionPosition(check6Press.keys))
            selectionIndicatorPrac.setAutoDraw(True)
            t_started = core.Clock().getTime()
        
        # *check6Press* updates
        waitOnFlip = False
        
        # if check6Press is starting this frame...
        if check6Press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check6Press.frameNStart = frameN  # exact frame index
            check6Press.tStart = t  # local t and not account for scr refresh
            check6Press.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check6Press, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check6Press.started')
            # update status
            check6Press.status = STARTED
            # allowed keys looks like a variable named `allowedKeys`
            if not type(allowedKeys) in [list, tuple, np.ndarray]:
                if not isinstance(allowedKeys, str):
                    allowedKeys = str(allowedKeys)
                elif not ',' in allowedKeys:
                    allowedKeys = (allowedKeys,)
                else:
                    allowedKeys = eval(allowedKeys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(check6Press.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(check6Press.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if check6Press.status == STARTED and not waitOnFlip:
            theseKeys = check6Press.getKeys(keyList=list(allowedKeys), ignoreKeys=["escape"], waitRelease=False)
            _check6Press_allKeys.extend(theseKeys)
            if len(_check6Press_allKeys):
                check6Press.keys = _check6Press_allKeys[-1].name  # just the last key pressed
                check6Press.rt = _check6Press_allKeys[-1].rt
                check6Press.duration = _check6Press_allKeys[-1].duration
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=check6,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            check6.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in check6.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "check6" ---
    for thisComponent in check6.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for check6
    check6.tStop = globalClock.getTime(format='float')
    check6.tStopRefresh = tThisFlipGlobal
    thisExp.addData('check6.stopped', check6.tStop)
    # Run 'End Routine' code from check6Select
    selectionIndicatorPrac.setAutoDraw(False)
    
    # check responses
    if check6Press.keys in ['', [], None]:  # No response was made
        check6Press.keys = None
    thisExp.addData('check6Press.keys',check6Press.keys)
    if check6Press.keys != None:  # we had a response
        thisExp.addData('check6Press.rt', check6Press.rt)
        thisExp.addData('check6Press.duration', check6Press.duration)
    thisExp.nextEntry()
    # the Routine "check6" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "check7" ---
    # create an object to store info about Routine check7
    check7 = data.Routine(
        name='check7',
        components=[check7Text, check7ImageLeft, check7ImageRight, check7Press],
    )
    check7.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    check7ImageLeft.setImage(lg1Img)
    check7ImageRight.setImage(lg2Img)
    # Run 'Begin Routine' code from check7Select
    # Initialize cue end time for cues and selection indicator.
    # If no response is made in this time, the trial ends.
    # Otherwise, cueEndTime is updated to response
    # time plus the selection duration.
    
    selectionIndicatorPrac.status = NOT_STARTED
    t_started = False
    # create starting attributes for check7Press
    check7Press.keys = []
    check7Press.rt = []
    _check7Press_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'allowedKeys' in globals():
        allowedKeys = globals()['allowedKeys']
    # store start times for check7
    check7.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    check7.tStart = globalClock.getTime(format='float')
    check7.status = STARTED
    thisExp.addData('check7.started', check7.tStart)
    check7.maxDuration = None
    # keep track of which components have finished
    check7Components = check7.components
    for thisComponent in check7.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "check7" ---
    check7.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *check7Text* updates
        
        # if check7Text is starting this frame...
        if check7Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check7Text.frameNStart = frameN  # exact frame index
            check7Text.tStart = t  # local t and not account for scr refresh
            check7Text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check7Text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check7Text.started')
            # update status
            check7Text.status = STARTED
            check7Text.setAutoDraw(True)
        
        # if check7Text is active this frame...
        if check7Text.status == STARTED:
            # update params
            pass
        
        # *check7ImageLeft* updates
        
        # if check7ImageLeft is starting this frame...
        if check7ImageLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check7ImageLeft.frameNStart = frameN  # exact frame index
            check7ImageLeft.tStart = t  # local t and not account for scr refresh
            check7ImageLeft.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check7ImageLeft, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check7ImageLeft.started')
            # update status
            check7ImageLeft.status = STARTED
            check7ImageLeft.setAutoDraw(True)
        
        # if check7ImageLeft is active this frame...
        if check7ImageLeft.status == STARTED:
            # update params
            pass
        
        # *check7ImageRight* updates
        
        # if check7ImageRight is starting this frame...
        if check7ImageRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check7ImageRight.frameNStart = frameN  # exact frame index
            check7ImageRight.tStart = t  # local t and not account for scr refresh
            check7ImageRight.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check7ImageRight, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check7ImageRight.started')
            # update status
            check7ImageRight.status = STARTED
            check7ImageRight.setAutoDraw(True)
        
        # if check7ImageRight is active this frame...
        if check7ImageRight.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from check7Select
        # If the selection has been drawn for the selection duration,
        # or the trial should end (cutting off the selection duration), end the routine
        if t_started:
            if t > t_started + 5:
                continueRoutine = False
        
        # Start Drawing Selection Indicator if Selection has been made
        if check7Press.keys and selectionIndicatorPrac.status == NOT_STARTED:
            #cueComponents.append(selectionIndicatorPrac)
            selectionIndicatorPrac.status = STARTED
            selectionIndicatorPrac.setPos(selectionPosition(check7Press.keys))
            selectionIndicatorPrac.setAutoDraw(True)
            t_started = core.Clock().getTime()
        
        # *check7Press* updates
        waitOnFlip = False
        
        # if check7Press is starting this frame...
        if check7Press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check7Press.frameNStart = frameN  # exact frame index
            check7Press.tStart = t  # local t and not account for scr refresh
            check7Press.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check7Press, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check7Press.started')
            # update status
            check7Press.status = STARTED
            # allowed keys looks like a variable named `allowedKeys`
            if not type(allowedKeys) in [list, tuple, np.ndarray]:
                if not isinstance(allowedKeys, str):
                    allowedKeys = str(allowedKeys)
                elif not ',' in allowedKeys:
                    allowedKeys = (allowedKeys,)
                else:
                    allowedKeys = eval(allowedKeys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(check7Press.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(check7Press.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if check7Press.status == STARTED and not waitOnFlip:
            theseKeys = check7Press.getKeys(keyList=list(allowedKeys), ignoreKeys=["escape"], waitRelease=False)
            _check7Press_allKeys.extend(theseKeys)
            if len(_check7Press_allKeys):
                check7Press.keys = _check7Press_allKeys[-1].name  # just the last key pressed
                check7Press.rt = _check7Press_allKeys[-1].rt
                check7Press.duration = _check7Press_allKeys[-1].duration
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=check7,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            check7.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in check7.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "check7" ---
    for thisComponent in check7.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for check7
    check7.tStop = globalClock.getTime(format='float')
    check7.tStopRefresh = tThisFlipGlobal
    thisExp.addData('check7.stopped', check7.tStop)
    # Run 'End Routine' code from check7Select
    selectionIndicatorPrac.setAutoDraw(False)
    
    # check responses
    if check7Press.keys in ['', [], None]:  # No response was made
        check7Press.keys = None
    thisExp.addData('check7Press.keys',check7Press.keys)
    if check7Press.keys != None:  # we had a response
        thisExp.addData('check7Press.rt', check7Press.rt)
        thisExp.addData('check7Press.duration', check7Press.duration)
    thisExp.nextEntry()
    # the Routine "check7" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "check8" ---
    # create an object to store info about Routine check8
    check8 = data.Routine(
        name='check8',
        components=[check8Text, check8ImageLeft, check8ImageRight, check8Press],
    )
    check8.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    check8ImageLeft.setImage(hl1Img)
    check8ImageRight.setImage(hl2Img)
    # Run 'Begin Routine' code from check8Select
    # Initialize cue end time for cues and selection indicator.
    # If no response is made in this time, the trial ends.
    # Otherwise, cueEndTime is updated to response
    # time plus the selection duration.
    
    selectionIndicatorPrac.status = NOT_STARTED
    t_started = False
    # create starting attributes for check8Press
    check8Press.keys = []
    check8Press.rt = []
    _check8Press_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'allowedKeys' in globals():
        allowedKeys = globals()['allowedKeys']
    # store start times for check8
    check8.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    check8.tStart = globalClock.getTime(format='float')
    check8.status = STARTED
    thisExp.addData('check8.started', check8.tStart)
    check8.maxDuration = None
    # keep track of which components have finished
    check8Components = check8.components
    for thisComponent in check8.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "check8" ---
    check8.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *check8Text* updates
        
        # if check8Text is starting this frame...
        if check8Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check8Text.frameNStart = frameN  # exact frame index
            check8Text.tStart = t  # local t and not account for scr refresh
            check8Text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check8Text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check8Text.started')
            # update status
            check8Text.status = STARTED
            check8Text.setAutoDraw(True)
        
        # if check8Text is active this frame...
        if check8Text.status == STARTED:
            # update params
            pass
        
        # *check8ImageLeft* updates
        
        # if check8ImageLeft is starting this frame...
        if check8ImageLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check8ImageLeft.frameNStart = frameN  # exact frame index
            check8ImageLeft.tStart = t  # local t and not account for scr refresh
            check8ImageLeft.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check8ImageLeft, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check8ImageLeft.started')
            # update status
            check8ImageLeft.status = STARTED
            check8ImageLeft.setAutoDraw(True)
        
        # if check8ImageLeft is active this frame...
        if check8ImageLeft.status == STARTED:
            # update params
            pass
        
        # *check8ImageRight* updates
        
        # if check8ImageRight is starting this frame...
        if check8ImageRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check8ImageRight.frameNStart = frameN  # exact frame index
            check8ImageRight.tStart = t  # local t and not account for scr refresh
            check8ImageRight.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check8ImageRight, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check8ImageRight.started')
            # update status
            check8ImageRight.status = STARTED
            check8ImageRight.setAutoDraw(True)
        
        # if check8ImageRight is active this frame...
        if check8ImageRight.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from check8Select
        # If the selection has been drawn for the selection duration,
        # or the trial should end (cutting off the selection duration), end the routine
        if t_started:
            if t > t_started + 5:
                continueRoutine = False
        
        # Start Drawing Selection Indicator if Selection has been made
        if check8Press.keys and selectionIndicatorPrac.status == NOT_STARTED:
            #cueComponents.append(selectionIndicatorPrac)
            selectionIndicatorPrac.status = STARTED
            selectionIndicatorPrac.setPos(selectionPosition(check8Press.keys))
            selectionIndicatorPrac.setAutoDraw(True)
            t_started = core.Clock().getTime()
        
        # *check8Press* updates
        waitOnFlip = False
        
        # if check8Press is starting this frame...
        if check8Press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check8Press.frameNStart = frameN  # exact frame index
            check8Press.tStart = t  # local t and not account for scr refresh
            check8Press.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check8Press, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check8Press.started')
            # update status
            check8Press.status = STARTED
            # allowed keys looks like a variable named `allowedKeys`
            if not type(allowedKeys) in [list, tuple, np.ndarray]:
                if not isinstance(allowedKeys, str):
                    allowedKeys = str(allowedKeys)
                elif not ',' in allowedKeys:
                    allowedKeys = (allowedKeys,)
                else:
                    allowedKeys = eval(allowedKeys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(check8Press.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(check8Press.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if check8Press.status == STARTED and not waitOnFlip:
            theseKeys = check8Press.getKeys(keyList=list(allowedKeys), ignoreKeys=["escape"], waitRelease=False)
            _check8Press_allKeys.extend(theseKeys)
            if len(_check8Press_allKeys):
                check8Press.keys = _check8Press_allKeys[-1].name  # just the last key pressed
                check8Press.rt = _check8Press_allKeys[-1].rt
                check8Press.duration = _check8Press_allKeys[-1].duration
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=check8,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            check8.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in check8.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "check8" ---
    for thisComponent in check8.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for check8
    check8.tStop = globalClock.getTime(format='float')
    check8.tStopRefresh = tThisFlipGlobal
    thisExp.addData('check8.stopped', check8.tStop)
    # Run 'End Routine' code from check8Select
    selectionIndicatorPrac.setAutoDraw(False)
    
    # check responses
    if check8Press.keys in ['', [], None]:  # No response was made
        check8Press.keys = None
    thisExp.addData('check8Press.keys',check8Press.keys)
    if check8Press.keys != None:  # we had a response
        thisExp.addData('check8Press.rt', check8Press.rt)
        thisExp.addData('check8Press.duration', check8Press.duration)
    thisExp.nextEntry()
    # the Routine "check8" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "check9" ---
    # create an object to store info about Routine check9
    check9 = data.Routine(
        name='check9',
        components=[check9Text, check9ImageLeft, check9ImageRight, check9Press],
    )
    check9.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    check9ImageLeft.setImage(ll1Img)
    check9ImageRight.setImage(ll2Img)
    # Run 'Begin Routine' code from check9Select
    # Initialize cue end time for cues and selection indicator.
    # If no response is made in this time, the trial ends.
    # Otherwise, cueEndTime is updated to response
    # time plus the selection duration.
    
    selectionIndicatorPrac.status = NOT_STARTED
    t_started = False
    # create starting attributes for check9Press
    check9Press.keys = []
    check9Press.rt = []
    _check9Press_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'allowedKeys' in globals():
        allowedKeys = globals()['allowedKeys']
    # store start times for check9
    check9.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    check9.tStart = globalClock.getTime(format='float')
    check9.status = STARTED
    thisExp.addData('check9.started', check9.tStart)
    check9.maxDuration = None
    # keep track of which components have finished
    check9Components = check9.components
    for thisComponent in check9.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "check9" ---
    check9.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *check9Text* updates
        
        # if check9Text is starting this frame...
        if check9Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check9Text.frameNStart = frameN  # exact frame index
            check9Text.tStart = t  # local t and not account for scr refresh
            check9Text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check9Text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check9Text.started')
            # update status
            check9Text.status = STARTED
            check9Text.setAutoDraw(True)
        
        # if check9Text is active this frame...
        if check9Text.status == STARTED:
            # update params
            pass
        
        # *check9ImageLeft* updates
        
        # if check9ImageLeft is starting this frame...
        if check9ImageLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check9ImageLeft.frameNStart = frameN  # exact frame index
            check9ImageLeft.tStart = t  # local t and not account for scr refresh
            check9ImageLeft.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check9ImageLeft, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check9ImageLeft.started')
            # update status
            check9ImageLeft.status = STARTED
            check9ImageLeft.setAutoDraw(True)
        
        # if check9ImageLeft is active this frame...
        if check9ImageLeft.status == STARTED:
            # update params
            pass
        
        # *check9ImageRight* updates
        
        # if check9ImageRight is starting this frame...
        if check9ImageRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check9ImageRight.frameNStart = frameN  # exact frame index
            check9ImageRight.tStart = t  # local t and not account for scr refresh
            check9ImageRight.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check9ImageRight, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check9ImageRight.started')
            # update status
            check9ImageRight.status = STARTED
            check9ImageRight.setAutoDraw(True)
        
        # if check9ImageRight is active this frame...
        if check9ImageRight.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from check9Select
        # If the selection has been drawn for the selection duration,
        # or the trial should end (cutting off the selection duration), end the routine
        if t_started:
            if t > t_started + 5:
                continueRoutine = False
        
        # Start Drawing Selection Indicator if Selection has been made
        if check9Press.keys and selectionIndicatorPrac.status == NOT_STARTED:
            #cueComponents.append(selectionIndicatorPrac)
            selectionIndicatorPrac.status = STARTED
            selectionIndicatorPrac.setPos(selectionPosition(check9Press.keys))
            selectionIndicatorPrac.setAutoDraw(True)
            t_started = core.Clock().getTime()
        
        # *check9Press* updates
        waitOnFlip = False
        
        # if check9Press is starting this frame...
        if check9Press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            check9Press.frameNStart = frameN  # exact frame index
            check9Press.tStart = t  # local t and not account for scr refresh
            check9Press.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(check9Press, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'check9Press.started')
            # update status
            check9Press.status = STARTED
            # allowed keys looks like a variable named `allowedKeys`
            if not type(allowedKeys) in [list, tuple, np.ndarray]:
                if not isinstance(allowedKeys, str):
                    allowedKeys = str(allowedKeys)
                elif not ',' in allowedKeys:
                    allowedKeys = (allowedKeys,)
                else:
                    allowedKeys = eval(allowedKeys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(check9Press.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(check9Press.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if check9Press.status == STARTED and not waitOnFlip:
            theseKeys = check9Press.getKeys(keyList=list(allowedKeys), ignoreKeys=["escape"], waitRelease=False)
            _check9Press_allKeys.extend(theseKeys)
            if len(_check9Press_allKeys):
                check9Press.keys = _check9Press_allKeys[-1].name  # just the last key pressed
                check9Press.rt = _check9Press_allKeys[-1].rt
                check9Press.duration = _check9Press_allKeys[-1].duration
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=check9,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            check9.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in check9.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "check9" ---
    for thisComponent in check9.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for check9
    check9.tStop = globalClock.getTime(format='float')
    check9.tStopRefresh = tThisFlipGlobal
    thisExp.addData('check9.stopped', check9.tStop)
    # Run 'End Routine' code from check9Select
    selectionIndicatorPrac.setAutoDraw(False)
    
    # check responses
    if check9Press.keys in ['', [], None]:  # No response was made
        check9Press.keys = None
    thisExp.addData('check9Press.keys',check9Press.keys)
    if check9Press.keys != None:  # we had a response
        thisExp.addData('check9Press.rt', check9Press.rt)
        thisExp.addData('check9Press.duration', check9Press.duration)
    thisExp.nextEntry()
    # the Routine "check9" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "done" ---
    # create an object to store info about Routine done
    done = data.Routine(
        name='done',
        components=[endExperiment, endTaskPress],
    )
    done.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for endTaskPress
    endTaskPress.keys = []
    endTaskPress.rt = []
    _endTaskPress_allKeys = []
    # store start times for done
    done.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    done.tStart = globalClock.getTime(format='float')
    done.status = STARTED
    thisExp.addData('done.started', done.tStart)
    done.maxDuration = None
    # keep track of which components have finished
    doneComponents = done.components
    for thisComponent in done.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "done" ---
    done.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *endExperiment* updates
        
        # if endExperiment is starting this frame...
        if endExperiment.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endExperiment.frameNStart = frameN  # exact frame index
            endExperiment.tStart = t  # local t and not account for scr refresh
            endExperiment.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endExperiment, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'endExperiment.started')
            # update status
            endExperiment.status = STARTED
            endExperiment.setAutoDraw(True)
        
        # if endExperiment is active this frame...
        if endExperiment.status == STARTED:
            # update params
            pass
        
        # *endTaskPress* updates
        waitOnFlip = False
        
        # if endTaskPress is starting this frame...
        if endTaskPress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endTaskPress.frameNStart = frameN  # exact frame index
            endTaskPress.tStart = t  # local t and not account for scr refresh
            endTaskPress.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endTaskPress, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'endTaskPress.started')
            # update status
            endTaskPress.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(endTaskPress.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(endTaskPress.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if endTaskPress.status == STARTED and not waitOnFlip:
            theseKeys = endTaskPress.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _endTaskPress_allKeys.extend(theseKeys)
            if len(_endTaskPress_allKeys):
                endTaskPress.keys = _endTaskPress_allKeys[-1].name  # just the last key pressed
                endTaskPress.rt = _endTaskPress_allKeys[-1].rt
                endTaskPress.duration = _endTaskPress_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=done,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            done.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in done.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "done" ---
    for thisComponent in done.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for done
    done.tStop = globalClock.getTime(format='float')
    done.tStopRefresh = tThisFlipGlobal
    thisExp.addData('done.stopped', done.tStop)
    # check responses
    if endTaskPress.keys in ['', [], None]:  # No response was made
        endTaskPress.keys = None
    thisExp.addData('endTaskPress.keys',endTaskPress.keys)
    if endTaskPress.keys != None:  # we had a response
        thisExp.addData('endTaskPress.rt', endTaskPress.rt)
        thisExp.addData('endTaskPress.duration', endTaskPress.duration)
    thisExp.nextEntry()
    # the Routine "done" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
