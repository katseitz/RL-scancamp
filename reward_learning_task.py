#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Mon Feb 17 14:49:16 2025
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
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

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
psychopyVersion = '2024.2.4'
expName = 'reward_learning_task'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'vers': 'A',
    'mriMode': 'Off',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
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
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/katharinaseitz/Documents/projects/reward_task/reward_learning_task.py',
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
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
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
    if deviceManager.getDevice('advanceScreenPress2') is None:
        # initialise advanceScreenPress2
        advanceScreenPress2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='advanceScreenPress2',
        )
    if deviceManager.getDevice('pracRespPress') is None:
        # initialise pracRespPress
        pracRespPress = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='pracRespPress',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
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
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
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
    if deviceManager.getDevice('endTaskPress') is None:
        # initialise endTaskPress
        endTaskPress = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='endTaskPress',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
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
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
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
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
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
    
    # --- Initialize components for Routine "pracDir" ---
    pracDirText = visual.TextStim(win=win, name='pracDirText',
        text='In this game, your job is to find the correct picture. Sometimes, you might have to guess. During the game, the correct picture might change. Try to choose the picture that is correct most of the time.\n\nThere will be two pictures on the screen, one on the left and one on the right. Press the button with your pointer finger to choose the shape on the left, and press the button with your middle finger to choose the shape on the right. The shapes will change sides, but this does not affect whether or not the shape is correct.\n\nMake your choice as fast as you can. Once you choose, a box will show up on the screen. If you choose too late, your choice will not count.\n\nAfter you make a choice, you will see a + on the screen for a few seconds. Next, you will get feedback telling you if you are correct or incorrect.',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    advanceScreenPress = keyboard.Keyboard(deviceName='advanceScreenPress')
    
    # --- Initialize components for Routine "pracDirections2" ---
    choiceDirText = visual.TextStim(win=win, name='choiceDirText',
        text='There will be two pictures on the screen, one on the left and one on the right. Press the button with your pointer finger to choose the picture on the left, and press the button with your middle finger to choose the picture on the right. The pictures will change sides, but this does not affect whether or not the picture is correct.\n\nMake your choice as fast as you can. Once you choose, a box will show up on the screen. If you choose too late, your choice will not count.\n\nPress the button now to make a choice.',
        font='Arial',
        pos=(0, 5), draggable=False, height=1.0, wrapWidth=40.0, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    advanceScreenPress2 = keyboard.Keyboard(deviceName='advanceScreenPress2')
    prac2Image = visual.ImageStim(
        win=win,
        name='prac2Image', 
        image='images/prac2.png', mask=None, anchor='center',
        ori=0.0, pos=(-10, -10), draggable=False, size=(10, 10),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    prac1Image = visual.ImageStim(
        win=win,
        name='prac1Image', 
        image='images/prac1.png', mask=None, anchor='center',
        ori=0.0, pos=(10, -10), draggable=False, size=(10, 10),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    selectionIndicator = visual.Rect(
        win=win, name='selectionIndicator',
        width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=-4.0, interpolate=True)
    pracRespPress = keyboard.Keyboard(deviceName='pracRespPress')
    
    # --- Initialize components for Routine "pracDirections3" ---
    text = visual.TextStim(win=win, name='text',
        text='After you make a choice, you will see a + on the screen for a few seconds. Next, you will get feedback telling you if you are correct or incorrect.\n\n\n\n\n\nDo you have any questions?\n\nPress space to start the practice.',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=30.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "gainDirs" ---
    advanceScreenPress3 = keyboard.Keyboard(deviceName='advanceScreenPress3')
    # Run 'Begin Experiment' code from setFrameColors
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
    frameDirText = visual.TextStim(win=win, name='frameDirText',
        text='Sometimes you can win money during the game!\n\nFor the HIGH WIN pair, you can win 50 cents if you are correct or win 0 cents if you are incorrect. \n\nFor the LOW WIN pair, you can win 25 cents if you are correct or win 0 cents if you are incorrect. \n\nThere will be a box around the pictures. The color of the box will tell you whether you can win a high or low amount. \n\nThe money you win will be paid to you as bonus at the end of the game, so try your best!',
        font='Arial',
        pos=(-5, 0), draggable=False, height=1.0, wrapWidth=20.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    highgainframe1_2 = visual.Rect(
        win=win, name='highgainframe1_2',
        width=(8, 6)[0], height=(8, 6)[1],
        ori=0.0, pos=(8, 8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=-3.0, interpolate=True)
    lowgainframe1_2 = visual.Rect(
        win=win, name='lowgainframe1_2',
        width=(8, 6)[0], height=(8, 6)[1],
        ori=0.0, pos=(8, -8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=-4.0, interpolate=True)
    highgainTopLabel_2 = visual.TextStim(win=win, name='highgainTopLabel_2',
        text='"+$0.50"',
        font='Arial',
        pos=(8.5, 8), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    lowgainTopLabel_2 = visual.TextStim(win=win, name='lowgainTopLabel_2',
        text='"+$0.25"',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    highgainBottomLabel_2 = visual.TextStim(win=win, name='highgainBottomLabel_2',
        text='"+$0.00"',
        font='Arial',
        pos=(8.5, 8), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    lowgainBottomLabel_2 = visual.TextStim(win=win, name='lowgainBottomLabel_2',
        text='"+$0.00"',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    
    # --- Initialize components for Routine "lossDirs" ---
    advanceScreen4 = keyboard.Keyboard(deviceName='advanceScreen4')
    lossFrameDirections = visual.TextStim(win=win, name='lossFrameDirections',
        text='Any text\n\nincluding line breaks',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "summaryDirections" ---
    summary = visual.TextStim(win=win, name='summary',
        text="Directions to remember:  \n\n1. Try to choose the picture that gives you the best chance of winning money and avoiding losing money. \n\n2. Press the left button with your POINTER finger to select the image on the left side of the screen. Press the right button with your MIDDLE finger to select the image on the right side of the screen.  \n\n3. The pictures will sometimes appear on opposite sides of the screen. This does not change whether they will win or lose.  \n\n4. Make your choice when you see the pictures. If you choose after that, your response won't be counted.  \n\n5. The money that you win in this task will be YOURS TO KEEP.'",
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=40.0, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from allowedKeys
    responseKeys = dict(left = '1', right = '2')
    allowedKeys = responseKeys.values()
    # Run 'Begin Experiment' code from conditionsFiles
    milConditionsFile1 = 'conditions/MIL_stakes_cond' + expInfo['vers'] + 's_block1.csv' 
    milConditionsFile2 = 'conditions/MIL_stakes_cond' + expInfo['vers'] + 's_block2.csv'
    milConditionsFile3 = 'conditions/MIL_stakes_cond' + expInfo['vers'] + 's_block3.csv'
    milConditionsFile4 = 'conditions/MIL_stakes_cond' + expInfo['vers'] + 's_block4.csv'
    
    logging.exp("Using conditions file: %s" % milConditionsFile1)
    advanceScreenPress4 = keyboard.Keyboard(deviceName='advanceScreenPress4')
    
    # --- Initialize components for Routine "getReady" ---
    getReadyText = visual.TextStim(win=win, name='getReadyText',
        text='Get Ready!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    advanceToTrigger = keyboard.Keyboard(deviceName='advanceToTrigger')
    
    # --- Initialize components for Routine "waitForScanner" ---
    # Run 'Begin Experiment' code from setRunFiles
    trialOrder = None
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Waiting for scanner (= remove this later)',
        font='Arial',
        pos=(0, 0), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    
    # --- Initialize components for Routine "cue" ---
    startFixStatic = visual.TextStim(win=win, name='startFixStatic',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=3.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from setCues
    import json
    jsonfile = 'Subject_Data/' + expInfo['participant'] + '.json'
    # Assign Cues to Condition or load previous set.
    if(os.path.exists(jsonfile)):
        with open(jsonfile,'r') as f:
            cuePairs = json.load(f)
    else:
        imageList = 'conditions/'+ 'PicList_'+ expInfo['vers'] + '.csv'
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
    
    
    # Run 'Begin Experiment' code from frameLabels
    topLabel=None
    bottomLabel=None
    framecolor=None
    frame1 = visual.Rect(
        win=win, name='frame1',
        width=(25,18)[0], height=(25,18)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=-3.0, interpolate=True)
    leftCue = visual.ImageStim(
        win=win,
        name='leftCue', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-10, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    rightCue = visual.ImageStim(
        win=win,
        name='rightCue', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(10, 0), draggable=False, size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    cueResp = keyboard.Keyboard(deviceName='cueResp')
    staticISI = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='staticISI')
    # Run 'Begin Experiment' code from manageTrialsLoop
    import time
    expInfo['expStartTime'] = time.ctime()
    
    TRIAL_DURATION = 10
    trialsClock = core.Clock()
    topFrameText = visual.TextStim(win=win, name='topFrameText',
        text=topLabel,
        font='Arial',
        pos=(0, 10), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    bottomFrameText = visual.TextStim(win=win, name='bottomFrameText',
        text=bottomLabel,
        font='Arial',
        pos=(0, -10), draggable=False, height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    # Run 'Begin Experiment' code from selectionIndicatorCode
    selectionDuration = 3
    
    def selectionPosition(response):
        '''Set position of selectionCue'''
        side = 1 if response == response_keys['right'] else -1  # For fMRI, button 1 == left and button 2 == right, so positive will shift right and negative will shift left
        return [side * 180,0]
    
    # Cue Choice Indicator
    selectionIndicator = visual.Polygon(win, edges =4, ori=45, radius=195, 
        name = 'selectionIndicator', lineColor = 'white', units ='pix', lineWidth=2, interpolate=True)
    
    #vertices=[[0,0], [0,260], [260,260], [260,0]]
    
    Choice=None
    
    # --- Initialize components for Routine "outcomeDelay" ---
    frame1Fix = visual.Rect(
        win=win, name='frame1Fix',
        width=(18, 10)[0], height=(18, 10)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=2.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    outcomeDelayFix = visual.TextStim(win=win, name='outcomeDelayFix',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    topFrameTextFix = visual.TextStim(win=win, name='topFrameTextFix',
        text='',
        font='Arial',
        pos=(0, 4.5), draggable=False, height=0.25, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    bottomFrameTextFix = visual.TextStim(win=win, name='bottomFrameTextFix',
        text='',
        font='Arial',
        pos=(0, -4.25), draggable=False, height=0.25, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "outcome" ---
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
        pos=(0, 0), draggable=False, height=0.4, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "fixation" ---
    fixationCrosshair = visual.TextStim(win=win, name='fixationCrosshair',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
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
    
    # --- Initialize components for Routine "Done" ---
    endExperiment = visual.TextStim(win=win, name='endExperiment',
        text='You finished the game!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.3, wrapWidth=None, ori=0.0, 
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
    
    # --- Prepare to start Routine "pracDir" ---
    # create an object to store info about Routine pracDir
    pracDir = data.Routine(
        name='pracDir',
        components=[pracDirText, advanceScreenPress],
    )
    pracDir.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for advanceScreenPress
    advanceScreenPress.keys = []
    advanceScreenPress.rt = []
    _advanceScreenPress_allKeys = []
    # store start times for pracDir
    pracDir.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    pracDir.tStart = globalClock.getTime(format='float')
    pracDir.status = STARTED
    thisExp.addData('pracDir.started', pracDir.tStart)
    pracDir.maxDuration = None
    # keep track of which components have finished
    pracDirComponents = pracDir.components
    for thisComponent in pracDir.components:
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
    
    # --- Run Routine "pracDir" ---
    pracDir.forceEnded = routineForceEnded = not continueRoutine
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
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            pracDir.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pracDir.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pracDir" ---
    for thisComponent in pracDir.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for pracDir
    pracDir.tStop = globalClock.getTime(format='float')
    pracDir.tStopRefresh = tThisFlipGlobal
    thisExp.addData('pracDir.stopped', pracDir.tStop)
    # check responses
    if advanceScreenPress.keys in ['', [], None]:  # No response was made
        advanceScreenPress.keys = None
    thisExp.addData('advanceScreenPress.keys',advanceScreenPress.keys)
    if advanceScreenPress.keys != None:  # we had a response
        thisExp.addData('advanceScreenPress.rt', advanceScreenPress.rt)
        thisExp.addData('advanceScreenPress.duration', advanceScreenPress.duration)
    thisExp.nextEntry()
    # the Routine "pracDir" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    pracBlock = data.TrialHandler2(
        name='pracBlock',
        nReps=1.0, 
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
        currentLoop = pracBlock
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPracBlock.rgb)
        if thisPracBlock != None:
            for paramName in thisPracBlock:
                globals()[paramName] = thisPracBlock[paramName]
        
        # --- Prepare to start Routine "pracDirections2" ---
        # create an object to store info about Routine pracDirections2
        pracDirections2 = data.Routine(
            name='pracDirections2',
            components=[choiceDirText, advanceScreenPress2, prac2Image, prac1Image, selectionIndicator, pracRespPress],
        )
        pracDirections2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for advanceScreenPress2
        advanceScreenPress2.keys = []
        advanceScreenPress2.rt = []
        _advanceScreenPress2_allKeys = []
        # create starting attributes for pracRespPress
        pracRespPress.keys = []
        pracRespPress.rt = []
        _pracRespPress_allKeys = []
        # Run 'Begin Routine' code from selectionBox
        cueEndTime = 20
        selectionIndicator.status = NOT_STARTED
        # store start times for pracDirections2
        pracDirections2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pracDirections2.tStart = globalClock.getTime(format='float')
        pracDirections2.status = STARTED
        thisExp.addData('pracDirections2.started', pracDirections2.tStart)
        pracDirections2.maxDuration = None
        # keep track of which components have finished
        pracDirections2Components = pracDirections2.components
        for thisComponent in pracDirections2.components:
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
        
        # --- Run Routine "pracDirections2" ---
        # if trial has changed, end Routine now
        if isinstance(pracBlock, data.TrialHandler2) and thisPracBlock.thisN != pracBlock.thisTrial.thisN:
            continueRoutine = False
        pracDirections2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
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
            
            # *advanceScreenPress2* updates
            waitOnFlip = False
            
            # if advanceScreenPress2 is starting this frame...
            if advanceScreenPress2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                advanceScreenPress2.frameNStart = frameN  # exact frame index
                advanceScreenPress2.tStart = t  # local t and not account for scr refresh
                advanceScreenPress2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(advanceScreenPress2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'advanceScreenPress2.started')
                # update status
                advanceScreenPress2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(advanceScreenPress2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(advanceScreenPress2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if advanceScreenPress2.status == STARTED and not waitOnFlip:
                theseKeys = advanceScreenPress2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _advanceScreenPress2_allKeys.extend(theseKeys)
                if len(_advanceScreenPress2_allKeys):
                    advanceScreenPress2.keys = _advanceScreenPress2_allKeys[-1].name  # just the last key pressed
                    advanceScreenPress2.rt = _advanceScreenPress2_allKeys[-1].rt
                    advanceScreenPress2.duration = _advanceScreenPress2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
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
            
            # *selectionIndicator* updates
            
            # if selectionIndicator is starting this frame...
            if selectionIndicator.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                selectionIndicator.frameNStart = frameN  # exact frame index
                selectionIndicator.tStart = t  # local t and not account for scr refresh
                selectionIndicator.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(selectionIndicator, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'selectionIndicator.started')
                # update status
                selectionIndicator.status = STARTED
                selectionIndicator.setAutoDraw(True)
            
            # if selectionIndicator is active this frame...
            if selectionIndicator.status == STARTED:
                # update params
                pass
            
            # *pracRespPress* updates
            waitOnFlip = False
            
            # if pracRespPress is starting this frame...
            if pracRespPress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pracRespPress.frameNStart = frameN  # exact frame index
                pracRespPress.tStart = t  # local t and not account for scr refresh
                pracRespPress.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pracRespPress, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pracRespPress.started')
                # update status
                pracRespPress.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(pracRespPress.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(pracRespPress.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if pracRespPress is stopping this frame...
            if pracRespPress.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > pracRespPress.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    pracRespPress.tStop = t  # not accounting for scr refresh
                    pracRespPress.tStopRefresh = tThisFlipGlobal  # on global time
                    pracRespPress.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pracRespPress.stopped')
                    # update status
                    pracRespPress.status = FINISHED
                    pracRespPress.status = FINISHED
            if pracRespPress.status == STARTED and not waitOnFlip:
                theseKeys = pracRespPress.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                _pracRespPress_allKeys.extend(theseKeys)
                if len(_pracRespPress_allKeys):
                    pracRespPress.keys = _pracRespPress_allKeys[0].name  # just the first key pressed
                    pracRespPress.rt = _pracRespPress_allKeys[0].rt
                    pracRespPress.duration = _pracRespPress_allKeys[0].duration
            # Run 'Each Frame' code from selectionBox
            # If the selection has been drawn for the seelection duration,
            # or if the trial should end (cutting of the selection duration)
            # end the routine
            
            if t > cueEndTime:
                continueRoutine = False
            # Start drawing the selection indicator if the 
            # selection has been made
            
            if pracRespPress.keys and selectionIndicator.status == NOT_STARTED:
                pracDirections2Components.append(selectionIndicator)
                slectionIndicator.status = STARTED
                selectionIndicator.setPost(selectionPositions(pracResp_dirs.keys))
                selectionIndicator.setAutoDraw(True)
            
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
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                pracDirections2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pracDirections2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pracDirections2" ---
        for thisComponent in pracDirections2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pracDirections2
        pracDirections2.tStop = globalClock.getTime(format='float')
        pracDirections2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pracDirections2.stopped', pracDirections2.tStop)
        # check responses
        if advanceScreenPress2.keys in ['', [], None]:  # No response was made
            advanceScreenPress2.keys = None
        pracBlock.addData('advanceScreenPress2.keys',advanceScreenPress2.keys)
        if advanceScreenPress2.keys != None:  # we had a response
            pracBlock.addData('advanceScreenPress2.rt', advanceScreenPress2.rt)
            pracBlock.addData('advanceScreenPress2.duration', advanceScreenPress2.duration)
        # check responses
        if pracRespPress.keys in ['', [], None]:  # No response was made
            pracRespPress.keys = None
        pracBlock.addData('pracRespPress.keys',pracRespPress.keys)
        if pracRespPress.keys != None:  # we had a response
            pracBlock.addData('pracRespPress.rt', pracRespPress.rt)
            pracBlock.addData('pracRespPress.duration', pracRespPress.duration)
        # the Routine "pracDirections2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "pracDirections3" ---
        # create an object to store info about Routine pracDirections3
        pracDirections3 = data.Routine(
            name='pracDirections3',
            components=[text, key_resp],
        )
        pracDirections3.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for pracDirections3
        pracDirections3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pracDirections3.tStart = globalClock.getTime(format='float')
        pracDirections3.status = STARTED
        thisExp.addData('pracDirections3.started', pracDirections3.tStart)
        pracDirections3.maxDuration = None
        # keep track of which components have finished
        pracDirections3Components = pracDirections3.components
        for thisComponent in pracDirections3.components:
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
        
        # --- Run Routine "pracDirections3" ---
        # if trial has changed, end Routine now
        if isinstance(pracBlock, data.TrialHandler2) and thisPracBlock.thisN != pracBlock.thisTrial.thisN:
            continueRoutine = False
        pracDirections3.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
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
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                pracDirections3.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pracDirections3.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pracDirections3" ---
        for thisComponent in pracDirections3.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pracDirections3
        pracDirections3.tStop = globalClock.getTime(format='float')
        pracDirections3.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pracDirections3.stopped', pracDirections3.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        pracBlock.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            pracBlock.addData('key_resp.rt', key_resp.rt)
            pracBlock.addData('key_resp.duration', key_resp.duration)
        # the Routine "pracDirections3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "gainDirs" ---
        # create an object to store info about Routine gainDirs
        gainDirs = data.Routine(
            name='gainDirs',
            components=[advanceScreenPress3, frameDirText, highgainframe1_2, lowgainframe1_2, highgainTopLabel_2, lowgainTopLabel_2, highgainBottomLabel_2, lowgainBottomLabel_2],
        )
        gainDirs.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for advanceScreenPress3
        advanceScreenPress3.keys = []
        advanceScreenPress3.rt = []
        _advanceScreenPress3_allKeys = []
        highgainframe1_2.setLineColor(highgainColor)
        lowgainframe1_2.setLineColor(lowgainColor)
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
        # if trial has changed, end Routine now
        if isinstance(pracBlock, data.TrialHandler2) and thisPracBlock.thisN != pracBlock.thisTrial.thisN:
            continueRoutine = False
        gainDirs.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
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
            
            # *frameDirText* updates
            
            # if frameDirText is starting this frame...
            if frameDirText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                frameDirText.frameNStart = frameN  # exact frame index
                frameDirText.tStart = t  # local t and not account for scr refresh
                frameDirText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(frameDirText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'frameDirText.started')
                # update status
                frameDirText.status = STARTED
                frameDirText.setAutoDraw(True)
            
            # if frameDirText is active this frame...
            if frameDirText.status == STARTED:
                # update params
                pass
            
            # *highgainframe1_2* updates
            
            # if highgainframe1_2 is starting this frame...
            if highgainframe1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                highgainframe1_2.frameNStart = frameN  # exact frame index
                highgainframe1_2.tStart = t  # local t and not account for scr refresh
                highgainframe1_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(highgainframe1_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'highgainframe1_2.started')
                # update status
                highgainframe1_2.status = STARTED
                highgainframe1_2.setAutoDraw(True)
            
            # if highgainframe1_2 is active this frame...
            if highgainframe1_2.status == STARTED:
                # update params
                pass
            
            # *lowgainframe1_2* updates
            
            # if lowgainframe1_2 is starting this frame...
            if lowgainframe1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lowgainframe1_2.frameNStart = frameN  # exact frame index
                lowgainframe1_2.tStart = t  # local t and not account for scr refresh
                lowgainframe1_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lowgainframe1_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lowgainframe1_2.started')
                # update status
                lowgainframe1_2.status = STARTED
                lowgainframe1_2.setAutoDraw(True)
            
            # if lowgainframe1_2 is active this frame...
            if lowgainframe1_2.status == STARTED:
                # update params
                pass
            
            # *highgainTopLabel_2* updates
            
            # if highgainTopLabel_2 is starting this frame...
            if highgainTopLabel_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                highgainTopLabel_2.frameNStart = frameN  # exact frame index
                highgainTopLabel_2.tStart = t  # local t and not account for scr refresh
                highgainTopLabel_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(highgainTopLabel_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'highgainTopLabel_2.started')
                # update status
                highgainTopLabel_2.status = STARTED
                highgainTopLabel_2.setAutoDraw(True)
            
            # if highgainTopLabel_2 is active this frame...
            if highgainTopLabel_2.status == STARTED:
                # update params
                pass
            
            # *lowgainTopLabel_2* updates
            
            # if lowgainTopLabel_2 is starting this frame...
            if lowgainTopLabel_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lowgainTopLabel_2.frameNStart = frameN  # exact frame index
                lowgainTopLabel_2.tStart = t  # local t and not account for scr refresh
                lowgainTopLabel_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lowgainTopLabel_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lowgainTopLabel_2.started')
                # update status
                lowgainTopLabel_2.status = STARTED
                lowgainTopLabel_2.setAutoDraw(True)
            
            # if lowgainTopLabel_2 is active this frame...
            if lowgainTopLabel_2.status == STARTED:
                # update params
                pass
            
            # *highgainBottomLabel_2* updates
            
            # if highgainBottomLabel_2 is starting this frame...
            if highgainBottomLabel_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                highgainBottomLabel_2.frameNStart = frameN  # exact frame index
                highgainBottomLabel_2.tStart = t  # local t and not account for scr refresh
                highgainBottomLabel_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(highgainBottomLabel_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'highgainBottomLabel_2.started')
                # update status
                highgainBottomLabel_2.status = STARTED
                highgainBottomLabel_2.setAutoDraw(True)
            
            # if highgainBottomLabel_2 is active this frame...
            if highgainBottomLabel_2.status == STARTED:
                # update params
                pass
            
            # *lowgainBottomLabel_2* updates
            
            # if lowgainBottomLabel_2 is starting this frame...
            if lowgainBottomLabel_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lowgainBottomLabel_2.frameNStart = frameN  # exact frame index
                lowgainBottomLabel_2.tStart = t  # local t and not account for scr refresh
                lowgainBottomLabel_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lowgainBottomLabel_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lowgainBottomLabel_2.started')
                # update status
                lowgainBottomLabel_2.status = STARTED
                lowgainBottomLabel_2.setAutoDraw(True)
            
            # if lowgainBottomLabel_2 is active this frame...
            if lowgainBottomLabel_2.status == STARTED:
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
                    timers=[routineTimer], 
                    playbackComponents=[]
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
            components=[advanceScreen4, lossFrameDirections],
        )
        lossDirs.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for advanceScreen4
        advanceScreen4.keys = []
        advanceScreen4.rt = []
        _advanceScreen4_allKeys = []
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
        # if trial has changed, end Routine now
        if isinstance(pracBlock, data.TrialHandler2) and thisPracBlock.thisN != pracBlock.thisTrial.thisN:
            continueRoutine = False
        lossDirs.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
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
            
            # *lossFrameDirections* updates
            
            # if lossFrameDirections is starting this frame...
            if lossFrameDirections.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                lossFrameDirections.frameNStart = frameN  # exact frame index
                lossFrameDirections.tStart = t  # local t and not account for scr refresh
                lossFrameDirections.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(lossFrameDirections, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'lossFrameDirections.started')
                # update status
                lossFrameDirections.status = STARTED
                lossFrameDirections.setAutoDraw(True)
            
            # if lossFrameDirections is active this frame...
            if lossFrameDirections.status == STARTED:
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
                    timers=[routineTimer], 
                    playbackComponents=[]
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
        # the Routine "lossDirs" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'pracBlock'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
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
                timers=[routineTimer], 
                playbackComponents=[]
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
        nReps=5.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
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
        # if trial has changed, end Routine now
        if isinstance(runs, data.TrialHandler2) and thisRun.thisN != runs.thisTrial.thisN:
            continueRoutine = False
        getReady.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
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
            
            # if getReadyText is stopping this frame...
            if getReadyText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > getReadyText.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    getReadyText.tStop = t  # not accounting for scr refresh
                    getReadyText.tStopRefresh = tThisFlipGlobal  # on global time
                    getReadyText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'getReadyText.stopped')
                    # update status
                    getReadyText.status = FINISHED
                    getReadyText.setAutoDraw(False)
            
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
                    timers=[routineTimer], 
                    playbackComponents=[]
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
            components=[text_2, key_resp_2, key_resp_3],
        )
        waitForScanner.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from setRunFiles
        if thisRun == 'Run1':
            trialOrder = milConditionsFile1
        if thisRun == 'Run2':
            trialOrder = milConditionsFile2
        if thisRun == 'Run3':
            trialOrder = milConditionsFile3
        if thisRun == 'Run4':
            trialOrder = milConditionsFile4
        # create starting attributes for key_resp_2
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
        # create starting attributes for key_resp_3
        key_resp_3.keys = []
        key_resp_3.rt = []
        _key_resp_3_allKeys = []
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
        # if trial has changed, end Routine now
        if isinstance(runs, data.TrialHandler2) and thisRun.thisN != runs.thisTrial.thisN:
            continueRoutine = False
        waitForScanner.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # *key_resp_2* updates
            waitOnFlip = False
            
            # if key_resp_2 is starting this frame...
            if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_2.frameNStart = frameN  # exact frame index
                key_resp_2.tStart = t  # local t and not account for scr refresh
                key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_2.started')
                # update status
                key_resp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_2.getKeys(keyList=['='], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_2_allKeys.extend(theseKeys)
                if len(_key_resp_2_allKeys):
                    key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                    key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                    key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *key_resp_3* updates
            waitOnFlip = False
            
            # if key_resp_3 is starting this frame...
            if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_3.frameNStart = frameN  # exact frame index
                key_resp_3.tStart = t  # local t and not account for scr refresh
                key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_3.started')
                # update status
                key_resp_3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_3.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_3.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_3_allKeys.extend(theseKeys)
                if len(_key_resp_3_allKeys):
                    key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                    key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                    key_resp_3.duration = _key_resp_3_allKeys[-1].duration
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
                    timers=[routineTimer], 
                    playbackComponents=[]
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
        # check responses
        if key_resp_2.keys in ['', [], None]:  # No response was made
            key_resp_2.keys = None
        runs.addData('key_resp_2.keys',key_resp_2.keys)
        if key_resp_2.keys != None:  # we had a response
            runs.addData('key_resp_2.rt', key_resp_2.rt)
            runs.addData('key_resp_2.duration', key_resp_2.duration)
        # check responses
        if key_resp_3.keys in ['', [], None]:  # No response was made
            key_resp_3.keys = None
        runs.addData('key_resp_3.keys',key_resp_3.keys)
        if key_resp_3.keys != None:  # we had a response
            runs.addData('key_resp_3.rt', key_resp_3.rt)
            runs.addData('key_resp_3.duration', key_resp_3.duration)
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
                components=[startFixStatic, frame1, leftCue, rightCue, cueResp, staticISI, topFrameText, bottomFrameText],
            )
            cue.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from setCues
            # update component parameters for each repeat
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
            # Run 'Begin Routine' code from frameLabels
            if condition == "lowgain":
                corrvalue=0.25
                incorrvalue=0
                topLabel="+$%.2f"%(abs(corrvalue))
                bottomLabel="+$%.2f"%(abs(incorrvalue))
            
            if condition == "highgain":
                corrvalue=0.50
                incorrvalue=0
                topLabel="+$%.2f"%(abs(corrvalue))
                bottomLabel="+$%.2f"%(abs(incorrvalue))
            
            if condition == "lowloss":
                corrvalue=0
                incorrvalue=-0.25
                topLabel="-$%.2f"%(abs(corrvalue))
                bottomLabel="-$%.2f"%(abs(incorrvalue))
            
            if condition == "highloss":
                corrvalue=0
                incorrvalue=-0.5
                topLabel="-$%.2f"%(abs(corrvalue))
                bottomLabel="-$%.2f"%(abs(incorrvalue))
            # Run 'Begin Routine' code from setImageTiming
            cueTimeAdded = False
            responseTimeAdded = False
            feedbackTimeAdded = False
            currentLoop.addData('startFixOnTime', fmriClock.getTime())
            # create starting attributes for cueResp
            cueResp.keys = []
            cueResp.rt = []
            _cueResp_allKeys = []
            # allowedKeys looks like a variable, so make sure it exists locally
            if 'allowed_keys' in globals():
                allowed_keys = globals()['allowed_keys']
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
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            cue.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 3.0:
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
                
                # *frame1* updates
                
                # if frame1 is starting this frame...
                if frame1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    frame1.frameNStart = frameN  # exact frame index
                    frame1.tStart = t  # local t and not account for scr refresh
                    frame1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(frame1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'frame1.started')
                    # update status
                    frame1.status = STARTED
                    frame1.setAutoDraw(True)
                
                # if frame1 is active this frame...
                if frame1.status == STARTED:
                    # update params
                    pass
                
                # if frame1 is stopping this frame...
                if frame1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > frame1.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        frame1.tStop = t  # not accounting for scr refresh
                        frame1.tStopRefresh = tThisFlipGlobal  # on global time
                        frame1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'frame1.stopped')
                        # update status
                        frame1.status = FINISHED
                        frame1.setAutoDraw(False)
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
                    # allowed keys looks like a variable named `allowed_keys`
                    if not type(allowed_keys) in [list, tuple, np.ndarray]:
                        if not isinstance(allowed_keys, str):
                            allowed_keys = str(allowed_keys)
                        elif not ',' in allowed_keys:
                            allowed_keys = (allowed_keys,)
                        else:
                            allowed_keys = eval(allowed_keys)
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
                    theseKeys = cueResp.getKeys(keyList=list(allowed_keys), ignoreKeys=["escape"], waitRelease=False)
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
                        # a response ends the routine
                        continueRoutine = False
                
                # *topFrameText* updates
                
                # if topFrameText is starting this frame...
                if topFrameText.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    topFrameText.frameNStart = frameN  # exact frame index
                    topFrameText.tStart = t  # local t and not account for scr refresh
                    topFrameText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(topFrameText, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'topFrameText.started')
                    # update status
                    topFrameText.status = STARTED
                    topFrameText.setAutoDraw(True)
                
                # if topFrameText is active this frame...
                if topFrameText.status == STARTED:
                    # update params
                    pass
                
                # if topFrameText is stopping this frame...
                if topFrameText.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > topFrameText.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        topFrameText.tStop = t  # not accounting for scr refresh
                        topFrameText.tStopRefresh = tThisFlipGlobal  # on global time
                        topFrameText.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'topFrameText.stopped')
                        # update status
                        topFrameText.status = FINISHED
                        topFrameText.setAutoDraw(False)
                
                # *bottomFrameText* updates
                
                # if bottomFrameText is starting this frame...
                if bottomFrameText.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    bottomFrameText.frameNStart = frameN  # exact frame index
                    bottomFrameText.tStart = t  # local t and not account for scr refresh
                    bottomFrameText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(bottomFrameText, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bottomFrameText.started')
                    # update status
                    bottomFrameText.status = STARTED
                    bottomFrameText.setAutoDraw(True)
                
                # if bottomFrameText is active this frame...
                if bottomFrameText.status == STARTED:
                    # update params
                    pass
                
                # if bottomFrameText is stopping this frame...
                if bottomFrameText.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > bottomFrameText.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        bottomFrameText.tStop = t  # not accounting for scr refresh
                        bottomFrameText.tStopRefresh = tThisFlipGlobal  # on global time
                        bottomFrameText.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'bottomFrameText.stopped')
                        # update status
                        bottomFrameText.status = FINISHED
                        bottomFrameText.setAutoDraw(False)
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
                    staticISI.start(0.5)
                elif staticISI.status == STARTED:  # one frame should pass before updating params and completing
                    staticISI.complete()  # finish the static period
                    staticISI.tStop = staticISI.tStart + 0.5  # record stop time
                
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
                        timers=[routineTimer], 
                        playbackComponents=[]
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
            # Run 'End Routine' code from setImageTiming
            currentLoop.addData('cueOffTime', fmriClock.getTime())
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
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if cue.maxDurationReached:
                routineTimer.addTime(-cue.maxDuration)
            elif cue.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-3.000000)
            
            # --- Prepare to start Routine "outcomeDelay" ---
            # create an object to store info about Routine outcomeDelay
            outcomeDelay = data.Routine(
                name='outcomeDelay',
                components=[frame1Fix, outcomeDelayFix, topFrameTextFix, bottomFrameTextFix],
            )
            outcomeDelay.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            topFrameTextFix.setText(topLabel)
            bottomFrameTextFix.setText(bottomLabel)
            # store start times for outcomeDelay
            outcomeDelay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            outcomeDelay.tStart = globalClock.getTime(format='float')
            outcomeDelay.status = STARTED
            thisExp.addData('outcomeDelay.started', outcomeDelay.tStart)
            outcomeDelay.maxDuration = None
            # keep track of which components have finished
            outcomeDelayComponents = outcomeDelay.components
            for thisComponent in outcomeDelay.components:
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
            
            # --- Run Routine "outcomeDelay" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            outcomeDelay.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *frame1Fix* updates
                
                # if frame1Fix is starting this frame...
                if frame1Fix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    frame1Fix.frameNStart = frameN  # exact frame index
                    frame1Fix.tStart = t  # local t and not account for scr refresh
                    frame1Fix.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(frame1Fix, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'frame1Fix.started')
                    # update status
                    frame1Fix.status = STARTED
                    frame1Fix.setAutoDraw(True)
                
                # if frame1Fix is active this frame...
                if frame1Fix.status == STARTED:
                    # update params
                    pass
                
                # if frame1Fix is stopping this frame...
                if frame1Fix.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > frame1Fix.tStartRefresh + outcomeDelay-frameTolerance:
                        # keep track of stop time/frame for later
                        frame1Fix.tStop = t  # not accounting for scr refresh
                        frame1Fix.tStopRefresh = tThisFlipGlobal  # on global time
                        frame1Fix.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'frame1Fix.stopped')
                        # update status
                        frame1Fix.status = FINISHED
                        frame1Fix.setAutoDraw(False)
                
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
                
                # *topFrameTextFix* updates
                
                # if topFrameTextFix is starting this frame...
                if topFrameTextFix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    topFrameTextFix.frameNStart = frameN  # exact frame index
                    topFrameTextFix.tStart = t  # local t and not account for scr refresh
                    topFrameTextFix.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(topFrameTextFix, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'topFrameTextFix.started')
                    # update status
                    topFrameTextFix.status = STARTED
                    topFrameTextFix.setAutoDraw(True)
                
                # if topFrameTextFix is active this frame...
                if topFrameTextFix.status == STARTED:
                    # update params
                    pass
                
                # if topFrameTextFix is stopping this frame...
                if topFrameTextFix.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > topFrameTextFix.tStartRefresh + outcomeDelay-frameTolerance:
                        # keep track of stop time/frame for later
                        topFrameTextFix.tStop = t  # not accounting for scr refresh
                        topFrameTextFix.tStopRefresh = tThisFlipGlobal  # on global time
                        topFrameTextFix.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'topFrameTextFix.stopped')
                        # update status
                        topFrameTextFix.status = FINISHED
                        topFrameTextFix.setAutoDraw(False)
                
                # *bottomFrameTextFix* updates
                
                # if bottomFrameTextFix is starting this frame...
                if bottomFrameTextFix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    bottomFrameTextFix.frameNStart = frameN  # exact frame index
                    bottomFrameTextFix.tStart = t  # local t and not account for scr refresh
                    bottomFrameTextFix.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(bottomFrameTextFix, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'bottomFrameTextFix.started')
                    # update status
                    bottomFrameTextFix.status = STARTED
                    bottomFrameTextFix.setAutoDraw(True)
                
                # if bottomFrameTextFix is active this frame...
                if bottomFrameTextFix.status == STARTED:
                    # update params
                    pass
                
                # if bottomFrameTextFix is stopping this frame...
                if bottomFrameTextFix.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > bottomFrameTextFix.tStartRefresh + outcomeDelay-frameTolerance:
                        # keep track of stop time/frame for later
                        bottomFrameTextFix.tStop = t  # not accounting for scr refresh
                        bottomFrameTextFix.tStopRefresh = tThisFlipGlobal  # on global time
                        bottomFrameTextFix.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'bottomFrameTextFix.stopped')
                        # update status
                        bottomFrameTextFix.status = FINISHED
                        bottomFrameTextFix.setAutoDraw(False)
                
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
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    outcomeDelay.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in outcomeDelay.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "outcomeDelay" ---
            for thisComponent in outcomeDelay.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for outcomeDelay
            outcomeDelay.tStop = globalClock.getTime(format='float')
            outcomeDelay.tStopRefresh = tThisFlipGlobal
            thisExp.addData('outcomeDelay.stopped', outcomeDelay.tStop)
            # the Routine "outcomeDelay" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "outcome" ---
            # create an object to store info about Routine outcome
            outcome = data.Routine(
                name='outcome',
                components=[outcomeText],
            )
            outcome.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from rightChoice
            if cueResp.keys==sidePositions[optimalImg]:
                Choice=1
            elif cueResp.keys==sidePositions[suboptimalImg]:
                Choice=2
            else: 
                Choice="NA"
                
            Accuracy=cueResp.corr
            
            currentLoop.addData('Choice', Choice)
            currentLoop.addData('Accuracy', Accuracy)
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
            
            outcomeText.setText(outcome_msg)
            # store start times for outcome
            outcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            outcome.tStart = globalClock.getTime(format='float')
            outcome.status = STARTED
            thisExp.addData('outcome.started', outcome.tStart)
            outcome.maxDuration = None
            # keep track of which components have finished
            outcomeComponents = outcome.components
            for thisComponent in outcome.components:
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
            
            # --- Run Routine "outcome" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            outcome.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
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
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    outcome.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in outcome.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "outcome" ---
            for thisComponent in outcome.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for outcome
            outcome.tStop = globalClock.getTime(format='float')
            outcome.tStopRefresh = tThisFlipGlobal
            thisExp.addData('outcome.stopped', outcome.tStop)
            # Run 'End Routine' code from defineOutcomes
            expInfo['bank'] = bank
            currentLoop.addData('feedbackOffTime', fmriClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if outcome.maxDurationReached:
                routineTimer.addTime(-outcome.maxDuration)
            elif outcome.forceEnded:
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
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            fixation.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
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
                        timers=[routineTimer], 
                        playbackComponents=[]
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
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
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
        # if trial has changed, end Routine now
        if isinstance(runs, data.TrialHandler2) and thisRun.thisN != runs.thisTrial.thisN:
            continueRoutine = False
        breakScreen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
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
                    timers=[routineTimer], 
                    playbackComponents=[]
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
        thisExp.nextEntry()
        
    # completed 5.0 repeats of 'runs'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Done" ---
    # create an object to store info about Routine Done
    Done = data.Routine(
        name='Done',
        components=[endExperiment, endTaskPress],
    )
    Done.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for endTaskPress
    endTaskPress.keys = []
    endTaskPress.rt = []
    _endTaskPress_allKeys = []
    # store start times for Done
    Done.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Done.tStart = globalClock.getTime(format='float')
    Done.status = STARTED
    thisExp.addData('Done.started', Done.tStart)
    Done.maxDuration = None
    # keep track of which components have finished
    DoneComponents = Done.components
    for thisComponent in Done.components:
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
    
    # --- Run Routine "Done" ---
    Done.forceEnded = routineForceEnded = not continueRoutine
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
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Done.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Done.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Done" ---
    for thisComponent in Done.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Done
    Done.tStop = globalClock.getTime(format='float')
    Done.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Done.stopped', Done.tStop)
    # check responses
    if endTaskPress.keys in ['', [], None]:  # No response was made
        endTaskPress.keys = None
    thisExp.addData('endTaskPress.keys',endTaskPress.keys)
    if endTaskPress.keys != None:  # we had a response
        thisExp.addData('endTaskPress.rt', endTaskPress.rt)
        thisExp.addData('endTaskPress.duration', endTaskPress.duration)
    thisExp.nextEntry()
    # the Routine "Done" was not non-slip safe, so reset the non-slip timer
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
