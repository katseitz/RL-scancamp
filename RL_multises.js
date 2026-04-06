/******************** 
 * Rl_Multises *
 ********************/

import { core, data, sound, util, visual, hardware } from './lib/psychojs-2025.1.1.js';
const { PsychoJS } = core;
const { TrialHandler, MultiStairHandler } = data;
const { Scheduler } = util;
//some handy aliases as in the psychopy scripts;
const { abs, sin, cos, PI: pi, sqrt } = Math;
const { round } = util;


// store info about the experiment session:
let expName = 'RL_multises';  // from the Builder filename that created this script
let expInfo = {
    'participant': ["1", "2", "3", "4", "5", "6", "7", "8", "OS-1", "OS-2", "OS-3", "OS-4", "OS-5", "OS-6", "OS-7", "OS-8", "pilot-1", "pilot-2", "pilot-3", "pilot-4", "pilot-5", "pilot-6", "pilot-7", "pilot-8"],
    'session': ["ses-1", "ses-2", "ses-3", "ses-4", "ses-5", "ses-6", "ses-7", "ses-8"],
    'startFromRun': ["1", "2"],
    'mode': ["scan", "orientation", "pilot"],
    'practice': ["no", "yes"],
};
let PILOTING = util.getUrlParameters().has('__pilotToken');

// Start code blocks for 'Before Experiment'
// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([-1,-1,-1]),
  units: 'deg',
  waitBlanking: true,
  backgroundImage: '',
  backgroundFit: 'none',
});
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); },flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(introDirRoutineBegin());
flowScheduler.add(introDirRoutineEachFrame());
flowScheduler.add(introDirRoutineEnd());
const pracBlockLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(pracBlockLoopBegin(pracBlockLoopScheduler));
flowScheduler.add(pracBlockLoopScheduler);
flowScheduler.add(pracBlockLoopEnd);





















flowScheduler.add(summaryDirectionsRoutineBegin());
flowScheduler.add(summaryDirectionsRoutineEachFrame());
flowScheduler.add(summaryDirectionsRoutineEnd());
const runsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(runsLoopBegin(runsLoopScheduler));
flowScheduler.add(runsLoopScheduler);
flowScheduler.add(runsLoopEnd);










flowScheduler.add(check6RoutineBegin());
flowScheduler.add(check6RoutineEachFrame());
flowScheduler.add(check6RoutineEnd());
flowScheduler.add(check7RoutineBegin());
flowScheduler.add(check7RoutineEachFrame());
flowScheduler.add(check7RoutineEnd());
flowScheduler.add(check8RoutineBegin());
flowScheduler.add(check8RoutineEachFrame());
flowScheduler.add(check8RoutineEnd());
flowScheduler.add(check9RoutineBegin());
flowScheduler.add(check9RoutineEachFrame());
flowScheduler.add(check9RoutineEnd());
flowScheduler.add(doneRoutineBegin());
flowScheduler.add(doneRoutineEachFrame());
flowScheduler.add(doneRoutineEnd());
flowScheduler.add(quitPsychoJS, 'Thank you for your patience.', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, 'Thank you for your patience.', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    // resources:
    {'name': 'conditions/practiceTrials.xlsx', 'path': 'conditions/practiceTrials.xlsx'},
    {'name': 'conditions/prac2Trials.csv', 'path': 'conditions/prac2Trials.csv'},
    {'name': 'conditions/Runs.xlsx', 'path': 'conditions/Runs.xlsx'},
    {'name': 'default.png', 'path': 'https://pavlovia.org/assets/default/default.png'},
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.INFO);

async function updateInfo() {
  currentLoop = psychoJS.experiment;  // right now there are no loops
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2025.1.1';
  expInfo['OS'] = window.navigator.platform;


  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  

  
  psychoJS.experiment.dataFileName = (("." + "/") + `data/${expInfo["participant"]}_${expName}_${expInfo["session"]}_${expInfo["date"]}`);
  psychoJS.experiment.field_separator = '\t';


  return Scheduler.Event.NEXT;
}

async function experimentInit() {
  // Initialize components for Routine "introDir"
  introDirClock = new util.Clock();
  pracDirText = new visual.TextStim({
    win: psychoJS.window,
    name: 'pracDirText',
    text: 'In this game, your job is to find the correct picture. Sometimes, you might have to guess. During the game, the correct picture might change. Try to choose the picture that is correct most of the time.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: 30.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: 0.0 
  });
  
  advanceScreenPress = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Run 'Begin Experiment' code from initializeVarCode
  /* Syntax Error: Fix Python code */
  // Initialize components for Routine "pracDir1"
  pracDir1Clock = new util.Clock();
  choiceDirText = new visual.TextStim({
    win: psychoJS.window,
    name: 'choiceDirText',
    text: 'There will be two pictures on the screen, one on the left and one on the right. Press the button with your pointer finger to choose the picture on the left, and press the button with your middle finger to choose the picture on the right. The pictures will change sides, but this does not affect whether or not the picture is correct.\n\n\n\n\n\n\n\n\n\n\nMake your choice as fast as you can. Once you choose, a box will show up on the screen. If you choose too late, your choice will not count.\n\nPractice selecting an image now by pressing with either finger. ',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: 30.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: 0.0 
  });
  
  prac2Image = new visual.ImageStim({
    win : psychoJS.window,
    name : 'prac2Image', units : undefined, 
    image : 'stimuli_sets/prac2.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 4), 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  prac1Image = new visual.ImageStim({
    win : psychoJS.window,
    name : 'prac1Image', units : undefined, 
    image : 'stimuli_sets/prac1.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [4, 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  pracPress = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "pracDir2"
  pracDir2Clock = new util.Clock();
  fixDirText = new visual.TextStim({
    win: psychoJS.window,
    name: 'fixDirText',
    text: 'After you make a choice, you will see a + on the screen for a few seconds. Next, you will get feedback telling you if you are correct or incorrect.\n\n\n\n\n\nDo you have any questions?\n\nPress space bar to start the practice.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: 30.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "pracCue"
  pracCueClock = new util.Clock();
  pracLeftImg = new visual.ImageStim({
    win : psychoJS.window,
    name : 'pracLeftImg', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 4), 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : 0.0 
  });
  pracRightImg = new visual.ImageStim({
    win : psychoJS.window,
    name : 'pracRightImg', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [4, 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  pracResp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Run 'Begin Experiment' code from setRepeatsCode
  if ((expInfo["practice"] === "yes")) {
      pracOn = 1;
  } else {
      pracOn = 0;
  }
  
  // Run 'Begin Experiment' code from selectionCode
  selectionDuration = 2.5;
  function selectionPosition(response) {
      /* Set position of selectionCue */
      var side;
      side = ((response === responseKeys["right"]) ? 0.75 : (- 0.75));
      return [(side * 180), 0];
  }
  selectionIndicatorPrac = new visual.Polygon(psychoJS.window, {"edges": 4, "ori": 45, "radius": 180, "name": "selectionIndicatorPrac", "lineColor": "white", "fillColor": null, "units": "pix", "lineWidth": 3, "interpolate": true});
  Choice = null;
  
  // Initialize components for Routine "pracJitter"
  pracJitterClock = new util.Clock();
  pracFix = new visual.TextStim({
    win: psychoJS.window,
    name: 'pracFix',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Run 'Begin Experiment' code from outcomeCode
  pracOutcome = null;
  
  // Initialize components for Routine "pracOut"
  pracOutClock = new util.Clock();
  pracOutText = new visual.TextStim({
    win: psychoJS.window,
    name: 'pracOutText',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  pracOutFix = new visual.TextStim({
    win: psychoJS.window,
    name: 'pracOutFix',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  // Initialize components for Routine "check1"
  check1Clock = new util.Clock();
  check1Text = new visual.TextStim({
    win: psychoJS.window,
    name: 'check1Text',
    text: "Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. If you don't know, press the down-arrow key.",
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: 30.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: 0.0 
  });
  
  check1ImageLeft = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check1ImageLeft', units : undefined, 
    image : 'stimuli_sets/prac2.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 4), 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  check1ImageRight = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check1ImageRight', units : undefined, 
    image : 'stimuli_sets/prac1.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [4, 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  check1Press = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  down1 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "phase2Dir"
  phase2DirClock = new util.Clock();
  phase2Instruc = new visual.TextStim({
    win: psychoJS.window,
    name: 'phase2Instruc',
    text: 'Now, when you play the game, you can win money. \n\nYou will see 4 different choice pairs. Each pair can win or lose a different amount. \n\nPress the space bar to proceed.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: 30.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  phase2Press = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "gainDirs"
  gainDirsClock = new util.Clock();
  advanceScreenPress3 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  gainDirText = new visual.TextStim({
    win: psychoJS.window,
    name: 'gainDirText',
    text: 'Sometimes you can win money during the game!\n\nFor the HIGH WIN pair, you can win $1.00 if you are correct or win $0.00 if you are incorrect. \n\nFor the LOW WIN pair, you can win $0.20 if you are correct or win $0.00 if you are incorrect. \n\nThere will be a box around the pictures. The numbers in the box will tell you whether you can win a high or low amount. \n\nThe money you win will be paid to you as bonus at the end of the game, so try your best!',
    font: 'Arial',
    units: undefined, 
    pos: [(- 6), 0], draggable: false, height: 1.0,  wrapWidth: 20.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  highgainframe = new visual.Rect ({
    win: psychoJS.window, name: 'highgainframe', 
    width: [8, 6][0], height: [8, 6][1],
    ori: 0.0, 
    pos: [10, 8], 
    draggable: false, 
    anchor: 'center', 
    lineWidth: 1.0, 
    lineColor: new util.Color('white'), 
    fillColor: undefined, 
    colorSpace: 'rgb', 
    opacity: undefined, 
    depth: -2, 
    interpolate: true, 
  });
  
  lowgainframe = new visual.Rect ({
    win: psychoJS.window, name: 'lowgainframe', 
    width: [8, 6][0], height: [8, 6][1],
    ori: 0.0, 
    pos: [10, (- 8)], 
    draggable: false, 
    anchor: 'center', 
    lineWidth: 1.0, 
    lineColor: new util.Color('white'), 
    fillColor: undefined, 
    colorSpace: 'rgb', 
    opacity: undefined, 
    depth: -3, 
    interpolate: true, 
  });
  
  highgainTopLabel = new visual.TextStim({
    win: psychoJS.window,
    name: 'highgainTopLabel',
    text: '+$1.00',
    font: 'Arial',
    units: undefined, 
    pos: [10, 10.5], draggable: false, height: 0.7,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -4.0 
  });
  
  lowgainTopLabel = new visual.TextStim({
    win: psychoJS.window,
    name: 'lowgainTopLabel',
    text: '+$0.20',
    font: 'Arial',
    units: undefined, 
    pos: [10, (- 5.5)], draggable: false, height: 0.7,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -5.0 
  });
  
  highgainBottomLabel = new visual.TextStim({
    win: psychoJS.window,
    name: 'highgainBottomLabel',
    text: '+$0.00',
    font: 'Arial',
    units: undefined, 
    pos: [10, 5.5], draggable: false, height: 0.7,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -6.0 
  });
  
  lowgainBottomLabel = new visual.TextStim({
    win: psychoJS.window,
    name: 'lowgainBottomLabel',
    text: '+$0.00',
    font: 'Arial',
    units: undefined, 
    pos: [10, (- 10.5)], draggable: false, height: 0.7,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -7.0 
  });
  
  highGainLabel = new visual.TextStim({
    win: psychoJS.window,
    name: 'highGainLabel',
    text: 'HIGH GAIN',
    font: 'Arial',
    units: undefined, 
    pos: [10, 8], draggable: false, height: 1.0,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -8.0 
  });
  
  lowGainLabel = new visual.TextStim({
    win: psychoJS.window,
    name: 'lowGainLabel',
    text: 'LOW GAIN',
    font: 'Arial',
    units: undefined, 
    pos: [10, (- 8)], draggable: false, height: 1.0,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -9.0 
  });
  
  // Initialize components for Routine "lossDirs"
  lossDirsClock = new util.Clock();
  advanceScreen4 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  lossDirText = new visual.TextStim({
    win: psychoJS.window,
    name: 'lossDirText',
    text: 'Sometimes you can lose money during the game!\n\nFor the HIGH LOSE pair, you can lose $0.00 if you are correct or lose $1.00 if you are incorrect.\n\nFor the LOW LOSE pair, you can lose $0.00 if you are correct or lose $0.20 if you are incorrect.\n\nThere will be a box around the pictures. The numbers in the box will tell you whether you can lose a high or low amount.\n\nThe money you lose will be taken away from the bonus money you win at the end of the game, so try your best!',
    font: 'Arial',
    units: undefined, 
    pos: [(- 6), 0], draggable: false, height: 1.0,  wrapWidth: 20.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  highLossFrame = new visual.Rect ({
    win: psychoJS.window, name: 'highLossFrame', 
    width: [8, 6][0], height: [8, 6][1],
    ori: 0.0, 
    pos: [10, 8], 
    draggable: false, 
    anchor: 'center', 
    lineWidth: 1.0, 
    lineColor: new util.Color('white'), 
    fillColor: undefined, 
    colorSpace: 'rgb', 
    opacity: undefined, 
    depth: -2, 
    interpolate: true, 
  });
  
  lowLossFrame = new visual.Rect ({
    win: psychoJS.window, name: 'lowLossFrame', 
    width: [8, 6][0], height: [8, 6][1],
    ori: 0.0, 
    pos: [10, (- 8)], 
    draggable: false, 
    anchor: 'center', 
    lineWidth: 1.0, 
    lineColor: new util.Color('white'), 
    fillColor: undefined, 
    colorSpace: 'rgb', 
    opacity: undefined, 
    depth: -3, 
    interpolate: true, 
  });
  
  highLossTopLabel = new visual.TextStim({
    win: psychoJS.window,
    name: 'highLossTopLabel',
    text: '-$0.00',
    font: 'Arial',
    units: undefined, 
    pos: [10, 10.5], draggable: false, height: 0.7,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -4.0 
  });
  
  lowLossTopLabel = new visual.TextStim({
    win: psychoJS.window,
    name: 'lowLossTopLabel',
    text: '-$0.00',
    font: 'Arial',
    units: undefined, 
    pos: [10, (- 5.5)], draggable: false, height: 0.7,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -5.0 
  });
  
  highLossBottomLabel = new visual.TextStim({
    win: psychoJS.window,
    name: 'highLossBottomLabel',
    text: '-$1.00',
    font: 'Arial',
    units: undefined, 
    pos: [10, 5.5], draggable: false, height: 0.7,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -6.0 
  });
  
  lowLossBottomLabel = new visual.TextStim({
    win: psychoJS.window,
    name: 'lowLossBottomLabel',
    text: '-$0.20',
    font: 'Arial',
    units: undefined, 
    pos: [10, (- 10.5)], draggable: false, height: 0.7,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -7.0 
  });
  
  highLossLabel = new visual.TextStim({
    win: psychoJS.window,
    name: 'highLossLabel',
    text: 'HIGH LOSE',
    font: 'Arial',
    units: undefined, 
    pos: [10, 8], draggable: false, height: 1.0,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -8.0 
  });
  
  lowLossLabel = new visual.TextStim({
    win: psychoJS.window,
    name: 'lowLossLabel',
    text: 'LOW GAIN',
    font: 'Arial',
    units: undefined, 
    pos: [10, (- 8)], draggable: false, height: 1.0,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -9.0 
  });
  
  // Run 'Begin Experiment' code from framePracOn
  frame = new visual.Rect({"win": psychoJS.window, "name": "frame1", "width": [20, 13][0], "height": [20, 13][1], "ori": 0.0, "pos": [0, 0], "draggable": false, "anchor": "center", "lineWidth": 4.0, "colorSpace": "rgb", "lineColor": "white", "fillColor": null, "opacity": null, "depth": (- 3.0), "interpolate": true});
  
  // Initialize components for Routine "prac2Cue"
  prac2CueClock = new util.Clock();
  prac2LeftCue = new visual.ImageStim({
    win : psychoJS.window,
    name : 'prac2LeftCue', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 4), 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : 0.0 
  });
  prac2RightCue = new visual.ImageStim({
    win : psychoJS.window,
    name : 'prac2RightCue', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [4, 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  prac2Resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Run 'Begin Experiment' code from setCuesPrac
  cuePairPrac = {"highgain": {"optimalChoice": "prac1.png", "suboptimalChoice": "prac2.png"}, "lowgain": {"optimalChoice": "prac3.png", "suboptimalChoice": "prac4.png"}, "highloss": {"optimalChoice": "prac5.png", "suboptimalChoice": "prac6.png"}, "lowloss": {"optimalChoice": "prac7.png", "suboptimalChoice": "prac8.png"}};
  
  ISI = new core.MinimalStim({
    name: "ISI", 
    win: psychoJS.window,
    autoDraw: false, 
    autoLog: true, 
  });
  // Run 'Begin Experiment' code from pracFrameLabels
  topLabel = null;
  bottomLabel = null;
  framecolor = null;
  topFrameText = new visual.TextStim({"win": psychoJS.window, "name": "topFrameText", "text": "", "font": "Arial", "pos": [0, 5.5], "draggable": false, "height": 0.7, "wrapWidth": null, "ori": 0.0, "color": "white", "colorSpace": "rgb", "opacity": null, "languageStyle": "LTR", "depth": (- 10.0)});
  bottomFrameText = new visual.TextStim({"win": psychoJS.window, "name": "bottomFrameText", "text": "", "font": "Arial", "pos": [0, (- 5.5)], "draggable": false, "height": 0.7, "wrapWidth": null, "ori": 0.0, "color": "white", "colorSpace": "rgb", "opacity": null, "languageStyle": "LTR", "depth": (- 11.0)});
  
  // Initialize components for Routine "prac2Jitter"
  prac2JitterClock = new util.Clock();
  prac2Fix = new visual.TextStim({
    win: psychoJS.window,
    name: 'prac2Fix',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Initialize components for Routine "prac2Out"
  prac2OutClock = new util.Clock();
  // Run 'Begin Experiment' code from pracRightChoice
  Accuracy = null;
  
  // Run 'Begin Experiment' code from definePracOut
  bank = 0;
  function choose_prac_outcome_from_cue(response) {
      /*Choose an outcome to display (if True, then the trial action will occur, if False it won't)
      by setting a high or low threshold based on the response and image chosen, and then
      see if a random "roll" exceeds the threshold.
  
      The roll must be greater than threshold to return True, so a high probability will have a low
      threshold, and vice versa.*/
      if ((sidePositions[optimalImg] === prac2Resp.keys)) {
          return ((optcorrect === 1) ? true : false);
      } else {
          if ((sidePositions[suboptimalImg] === prac2Resp.keys)) {
              return ((optcorrect === 0) ? true : false);
          }
      }
  }
  Accuracy_OptChoice = null;
  
  outcomePracText = new visual.TextStim({
    win: psychoJS.window,
    name: 'outcomePracText',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.75,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -3.0 
  });
  
  // Initialize components for Routine "check3"
  check3Clock = new util.Clock();
  check3Text = new visual.TextStim({
    win: psychoJS.window,
    name: 'check3Text',
    text: "Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. If you don't know, press the down-arrow key.",
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: 30.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: 0.0 
  });
  
  check3ImageLeft = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check3ImageLeft', units : undefined, 
    image : 'stimuli_sets/prac3.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 4), 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  check3ImageRight = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check3ImageRight', units : undefined, 
    image : 'stimuli_sets/prac4.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [4, 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  check3Press = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  down2 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "check2"
  check2Clock = new util.Clock();
  check2Text = new visual.TextStim({
    win: psychoJS.window,
    name: 'check2Text',
    text: "Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. If you don't know, press the down-arrow key.",
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: 30.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: 0.0 
  });
  
  check2ImageLeft = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check2ImageLeft', units : undefined, 
    image : 'stimuli_sets/prac2.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 4), 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  check2ImageRight = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check2ImageRight', units : undefined, 
    image : 'stimuli_sets/prac1.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [4, 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  check2Press = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  down3 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "check4"
  check4Clock = new util.Clock();
  check4Text = new visual.TextStim({
    win: psychoJS.window,
    name: 'check4Text',
    text: "Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. If you don't know, press the down-arrow key.",
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: 30.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: 0.0 
  });
  
  check4ImageLeft = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check4ImageLeft', units : undefined, 
    image : 'stimuli_sets/prac5.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 4), 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  check4ImageRight = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check4ImageRight', units : undefined, 
    image : 'stimuli_sets/prac6.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [4, 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  check4Press = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  down4 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "check5"
  check5Clock = new util.Clock();
  check5Text = new visual.TextStim({
    win: psychoJS.window,
    name: 'check5Text',
    text: "Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. If you don't know, press the down-arrow key.",
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: 30.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: 0.0 
  });
  
  check5ImageLeft = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check5ImageLeft', units : undefined, 
    image : 'stimuli_sets/prac7.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 4), 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  check5ImageRight = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check5ImageRight', units : undefined, 
    image : 'stimuli_sets/prac8.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [4, 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  check5Press = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  down5 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "summaryDirections"
  summaryDirectionsClock = new util.Clock();
  summary = new visual.TextStim({
    win: psychoJS.window,
    name: 'summary',
    text: "Directions to remember:  \n\n1. Try to choose the picture that gives you the best chance of winning money and avoiding losing money. \n\n2. Press the left button with your POINTER finger to select the image on the left side of the screen. Press the right button with your MIDDLE finger to select the image on the right side of the screen.  \n\n3. The pictures will sometimes appear on opposite sides of the screen. This does not change whether they will win or lose.  \n\n4. Make your choice when you see the pictures. If you choose after that, your response won't be counted.  \n\n5. The money that you win in this task will be YOURS TO KEEP.",
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: 40.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Run 'Begin Experiment' code from conditionsFiles
  /* Syntax Error: Fix Python code */
  advanceScreenPress4 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "getReady"
  getReadyClock = new util.Clock();
  getReadyText = new visual.TextStim({
    win: psychoJS.window,
    name: 'getReadyText',
    text: 'Get Ready!',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  advanceToTrigger = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Run 'Begin Experiment' code from setRunFiles
  trialOrder = null;
  
  // Initialize components for Routine "waitForScanner"
  waitForScannerClock = new util.Clock();
  // Run 'Begin Experiment' code from waitScannerCode
  waitForScannerClock = new util.Clock();
  fmriClock = new util.Clock();
  trigger = "usb";
  /*
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
  }*/
  
  scannerTriggerKey = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Run 'Begin Experiment' code from frameOnCode
  frame = new visual.Rect({"win": psychoJS.window, "name": "frame1", "width": [20, 13][0], "height": [20, 13][1], "ori": 0.0, "pos": [0, 0], "draggable": false, "anchor": "center", "lineWidth": 4.0, "colorSpace": "rgb", "lineColor": "white", "fillColor": null, "opacity": null, "depth": (- 3.0), "interpolate": true});
  
  sumTemplate = new visual.ImageStim({
    win : psychoJS.window,
    name : 'sumTemplate', units : undefined, 
    image : 'summary.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0, 0], 
    draggable: false,
    size : [35, 18],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -3.0 
  });
  hg1 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'hg1', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 9.5), 1], 
    draggable: false,
    size : [2.5, 2.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -5.0 
  });
  hg2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'hg2', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 5.5), 1], 
    draggable: false,
    size : [2.5, 2.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -6.0 
  });
  lg1 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'lg1', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 9.5), (- 5.3)], 
    draggable: false,
    size : [2.5, 2.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -7.0 
  });
  lg2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'lg2', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 5.5), (- 5.3)], 
    draggable: false,
    size : [2.5, 2.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -8.0 
  });
  hl1 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'hl1', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [9.5, 1], 
    draggable: false,
    size : [2.5, 2.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -9.0 
  });
  hl2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'hl2', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [5.5, 1], 
    draggable: false,
    size : [2.5, 2.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -10.0 
  });
  ll1 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'll1', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [9.5, (- 5.3)], 
    draggable: false,
    size : [2.5, 2.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -11.0 
  });
  ll2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'll2', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [5.5, (- 5.3)], 
    draggable: false,
    size : [2.5, 2.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -12.0 
  });
  // Initialize components for Routine "cue"
  cueClock = new util.Clock();
  startFixStatic = new visual.TextStim({
    win: psychoJS.window,
    name: 'startFixStatic',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  leftCue = new visual.ImageStim({
    win : psychoJS.window,
    name : 'leftCue', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 4), 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  rightCue = new visual.ImageStim({
    win : psychoJS.window,
    name : 'rightCue', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [4, 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  cueResp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Run 'Begin Experiment' code from setCues
  /* Syntax Error: Fix Python code */
  // Run 'Begin Experiment' code from frameLabels
  topLabel = null;
  bottomLabel = null;
  framecolor = null;
  topFrameText = new visual.TextStim({"win": psychoJS.window, "name": "topFrameText", "text": "", "font": "Arial", "pos": [0, 5.5], "draggable": false, "height": 0.7, "wrapWidth": null, "ori": 0.0, "color": "white", "colorSpace": "rgb", "opacity": null, "languageStyle": "LTR", "depth": (- 10.0)});
  bottomFrameText = new visual.TextStim({"win": psychoJS.window, "name": "bottomFrameText", "text": "", "font": "Arial", "pos": [0, (- 5.5)], "draggable": false, "height": 0.7, "wrapWidth": null, "ori": 0.0, "color": "white", "colorSpace": "rgb", "opacity": null, "languageStyle": "LTR", "depth": (- 11.0)});
  
  staticISI = new core.MinimalStim({
    name: "staticISI", 
    win: psychoJS.window,
    autoDraw: false, 
    autoLog: true, 
  });
  // Run 'Begin Experiment' code from manageTrialsLoop
  import * as time from 'time';
  expInfo["expStartTime"] = time.ctime();
  TRIAL_DURATION = 10;
  trialsClock = new util.Clock();
  
  // Run 'Begin Experiment' code from selectionIndicatorCode
  selectionDuration = 3;
  function selectionPosition(response) {
      /* Set position of selectionCue */
      var side;
      side = ((response === responseKeys["right"]) ? 0.75 : (- 0.75));
      return [(side * 180), 0];
  }
  selectionIndicator = new visual.Polygon(psychoJS.window, {"edges": 4, "ori": 45, "radius": 180, "name": "selectionIndicator", "lineColor": "white", "fillColor": null, "units": "pix", "lineWidth": 3, "interpolate": true});
  Choice = null;
  
  // Initialize components for Routine "outcomeDelayRoutine"
  outcomeDelayRoutineClock = new util.Clock();
  outcomeDelayFix = new visual.TextStim({
    win: psychoJS.window,
    name: 'outcomeDelayFix',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Initialize components for Routine "outcomeRoutine"
  outcomeRoutineClock = new util.Clock();
  // Run 'Begin Experiment' code from rightChoice
  Accuracy = null;
  
  // Run 'Begin Experiment' code from defineOutcomes
  bank = 0;
  function choose_outcome_from_cue(response) {
      /*Choose an outcome to display (if True, then the trial action will occur, if False it won't)
      by setting a high or low threshold based on the response and image chosen, and then
      see if a random "roll" exceeds the threshold.
  
      The roll must be greater than threshold to return True, so a high probability will have a low
      threshold, and vice versa.*/
      if ((sidePositions[optimalImg] === cueResp.keys)) {
          return ((optcorrect === 1) ? true : false);
      } else {
          if ((sidePositions[suboptimalImg] === cueResp.keys)) {
              return ((optcorrect === 0) ? true : false);
          }
      }
  }
  Accuracy_OptChoice = null;
  
  outcomeText = new visual.TextStim({
    win: psychoJS.window,
    name: 'outcomeText',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.75,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -3.0 
  });
  
  // Initialize components for Routine "fixation"
  fixationClock = new util.Clock();
  fixationCrosshair = new visual.TextStim({
    win: psychoJS.window,
    name: 'fixationCrosshair',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Initialize components for Routine "breakScreen"
  breakScreenClock = new util.Clock();
  breakText = new visual.TextStim({
    win: psychoJS.window,
    name: 'breakText',
    text: 'You finished this round!',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  advanceScreenPress_2 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "check6"
  check6Clock = new util.Clock();
  check6Text = new visual.TextStim({
    win: psychoJS.window,
    name: 'check6Text',
    text: 'Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. ',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: 30.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: 0.0 
  });
  
  check6ImageLeft = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check6ImageLeft', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 4), 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  check6ImageRight = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check6ImageRight', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [4, 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  check6Press = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "check7"
  check7Clock = new util.Clock();
  check7Text = new visual.TextStim({
    win: psychoJS.window,
    name: 'check7Text',
    text: 'Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. ',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: 30.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: 0.0 
  });
  
  check7ImageLeft = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check7ImageLeft', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 4), 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  check7ImageRight = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check7ImageRight', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [4, 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  check7Press = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "check8"
  check8Clock = new util.Clock();
  check8Text = new visual.TextStim({
    win: psychoJS.window,
    name: 'check8Text',
    text: 'Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. ',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: 30.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: 0.0 
  });
  
  check8ImageLeft = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check8ImageLeft', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 4), 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  check8ImageRight = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check8ImageRight', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [4, 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  check8Press = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "check9"
  check9Clock = new util.Clock();
  check9Text = new visual.TextStim({
    win: psychoJS.window,
    name: 'check9Text',
    text: 'Which image was correct most of the time?\n\n\n\n\n\n\n\n\nSelect the image now. ',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: 30.0, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color([1.0, 1.0, 1.0]),  opacity: undefined,
    depth: 0.0 
  });
  
  check9ImageLeft = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check9ImageLeft', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 4), 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  check9ImageRight = new visual.ImageStim({
    win : psychoJS.window,
    name : 'check9ImageRight', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [4, 0], 
    draggable: false,
    size : [7, 7],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  check9Press = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "done"
  doneClock = new util.Clock();
  endExperiment = new visual.TextStim({
    win: psychoJS.window,
    name: 'endExperiment',
    text: 'You finished the game!',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 1.0,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  endTaskPress = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}

function introDirRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'introDir' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    introDirClock.reset();
    routineTimer.reset();
    introDirMaxDurationReached = false;
    // update component parameters for each repeat
    advanceScreenPress.keys = undefined;
    advanceScreenPress.rt = undefined;
    _advanceScreenPress_allKeys = [];
    psychoJS.experiment.addData('introDir.started', globalClock.getTime());
    introDirMaxDuration = null
    // keep track of which components have finished
    introDirComponents = [];
    introDirComponents.push(pracDirText);
    introDirComponents.push(advanceScreenPress);
    
    for (const thisComponent of introDirComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}

function introDirRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'introDir' ---
    // get current time
    t = introDirClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *pracDirText* updates
    if (t >= 0.0 && pracDirText.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pracDirText.tStart = t;  // (not accounting for frame time here)
      pracDirText.frameNStart = frameN;  // exact frame index
      
      pracDirText.setAutoDraw(true);
    }
    
    
    // if pracDirText is active this frame...
    if (pracDirText.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *advanceScreenPress* updates
    if (t >= 0.0 && advanceScreenPress.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      advanceScreenPress.tStart = t;  // (not accounting for frame time here)
      advanceScreenPress.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { advanceScreenPress.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { advanceScreenPress.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { advanceScreenPress.clearEvents(); });
    }
    
    // if advanceScreenPress is active this frame...
    if (advanceScreenPress.status === PsychoJS.Status.STARTED) {
      let theseKeys = advanceScreenPress.getKeys({keyList: 'space', waitRelease: false});
      _advanceScreenPress_allKeys = _advanceScreenPress_allKeys.concat(theseKeys);
      if (_advanceScreenPress_allKeys.length > 0) {
        advanceScreenPress.keys = _advanceScreenPress_allKeys[_advanceScreenPress_allKeys.length - 1].name;  // just the last key pressed
        advanceScreenPress.rt = _advanceScreenPress_allKeys[_advanceScreenPress_allKeys.length - 1].rt;
        advanceScreenPress.duration = _advanceScreenPress_allKeys[_advanceScreenPress_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      routineForceEnded = true;
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of introDirComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function introDirRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'introDir' ---
    for (const thisComponent of introDirComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('introDir.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(advanceScreenPress.corr, level);
    }
    psychoJS.experiment.addData('advanceScreenPress.keys', advanceScreenPress.keys);
    if (typeof advanceScreenPress.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('advanceScreenPress.rt', advanceScreenPress.rt);
        psychoJS.experiment.addData('advanceScreenPress.duration', advanceScreenPress.duration);
        routineTimer.reset();
        }
    
    advanceScreenPress.stop();
    // the Routine "introDir" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function pracBlockLoopBegin(pracBlockLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    pracBlock = new TrialHandler({
      psychoJS: psychoJS,
      nReps: pracBlockRepeats, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: undefined,
      seed: undefined, name: 'pracBlock'
    });
    psychoJS.experiment.addLoop(pracBlock); // add the loop to the experiment
    currentLoop = pracBlock;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisPracBlock of pracBlock) {
      snapshot = pracBlock.getSnapshot();
      pracBlockLoopScheduler.add(importConditions(snapshot));
      pracBlockLoopScheduler.add(pracDir1RoutineBegin(snapshot));
      pracBlockLoopScheduler.add(pracDir1RoutineEachFrame());
      pracBlockLoopScheduler.add(pracDir1RoutineEnd(snapshot));
      pracBlockLoopScheduler.add(pracDir2RoutineBegin(snapshot));
      pracBlockLoopScheduler.add(pracDir2RoutineEachFrame());
      pracBlockLoopScheduler.add(pracDir2RoutineEnd(snapshot));
      const practiceLoopScheduler = new Scheduler(psychoJS);
      pracBlockLoopScheduler.add(practiceLoopBegin(practiceLoopScheduler, snapshot));
      pracBlockLoopScheduler.add(practiceLoopScheduler);
      pracBlockLoopScheduler.add(practiceLoopEnd);
      pracBlockLoopScheduler.add(check1RoutineBegin(snapshot));
      pracBlockLoopScheduler.add(check1RoutineEachFrame());
      pracBlockLoopScheduler.add(check1RoutineEnd(snapshot));
      pracBlockLoopScheduler.add(phase2DirRoutineBegin(snapshot));
      pracBlockLoopScheduler.add(phase2DirRoutineEachFrame());
      pracBlockLoopScheduler.add(phase2DirRoutineEnd(snapshot));
      pracBlockLoopScheduler.add(gainDirsRoutineBegin(snapshot));
      pracBlockLoopScheduler.add(gainDirsRoutineEachFrame());
      pracBlockLoopScheduler.add(gainDirsRoutineEnd(snapshot));
      pracBlockLoopScheduler.add(lossDirsRoutineBegin(snapshot));
      pracBlockLoopScheduler.add(lossDirsRoutineEachFrame());
      pracBlockLoopScheduler.add(lossDirsRoutineEnd(snapshot));
      const practice2LoopScheduler = new Scheduler(psychoJS);
      pracBlockLoopScheduler.add(practice2LoopBegin(practice2LoopScheduler, snapshot));
      pracBlockLoopScheduler.add(practice2LoopScheduler);
      pracBlockLoopScheduler.add(practice2LoopEnd);
      pracBlockLoopScheduler.add(check3RoutineBegin(snapshot));
      pracBlockLoopScheduler.add(check3RoutineEachFrame());
      pracBlockLoopScheduler.add(check3RoutineEnd(snapshot));
      pracBlockLoopScheduler.add(check2RoutineBegin(snapshot));
      pracBlockLoopScheduler.add(check2RoutineEachFrame());
      pracBlockLoopScheduler.add(check2RoutineEnd(snapshot));
      pracBlockLoopScheduler.add(check4RoutineBegin(snapshot));
      pracBlockLoopScheduler.add(check4RoutineEachFrame());
      pracBlockLoopScheduler.add(check4RoutineEnd(snapshot));
      pracBlockLoopScheduler.add(check5RoutineBegin(snapshot));
      pracBlockLoopScheduler.add(check5RoutineEachFrame());
      pracBlockLoopScheduler.add(check5RoutineEnd(snapshot));
      pracBlockLoopScheduler.add(pracBlockLoopEndIteration(pracBlockLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}

function practiceLoopBegin(practiceLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    practice = new TrialHandler({
      psychoJS: psychoJS,
      nReps: pracOn, method: TrialHandler.Method.SEQUENTIAL,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'conditions/practiceTrials.xlsx',
      seed: undefined, name: 'practice'
    });
    psychoJS.experiment.addLoop(practice); // add the loop to the experiment
    currentLoop = practice;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisPractice of practice) {
      snapshot = practice.getSnapshot();
      practiceLoopScheduler.add(importConditions(snapshot));
      practiceLoopScheduler.add(pracCueRoutineBegin(snapshot));
      practiceLoopScheduler.add(pracCueRoutineEachFrame());
      practiceLoopScheduler.add(pracCueRoutineEnd(snapshot));
      practiceLoopScheduler.add(pracJitterRoutineBegin(snapshot));
      practiceLoopScheduler.add(pracJitterRoutineEachFrame());
      practiceLoopScheduler.add(pracJitterRoutineEnd(snapshot));
      practiceLoopScheduler.add(pracOutRoutineBegin(snapshot));
      practiceLoopScheduler.add(pracOutRoutineEachFrame());
      practiceLoopScheduler.add(pracOutRoutineEnd(snapshot));
      practiceLoopScheduler.add(practiceLoopEndIteration(practiceLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}

async function practiceLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(practice);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function practiceLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}

function practice2LoopBegin(practice2LoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    practice2 = new TrialHandler({
      psychoJS: psychoJS,
      nReps: pracOn, method: TrialHandler.Method.SEQUENTIAL,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'conditions/prac2Trials.csv',
      seed: undefined, name: 'practice2'
    });
    psychoJS.experiment.addLoop(practice2); // add the loop to the experiment
    currentLoop = practice2;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisPractice2 of practice2) {
      snapshot = practice2.getSnapshot();
      practice2LoopScheduler.add(importConditions(snapshot));
      practice2LoopScheduler.add(prac2CueRoutineBegin(snapshot));
      practice2LoopScheduler.add(prac2CueRoutineEachFrame());
      practice2LoopScheduler.add(prac2CueRoutineEnd(snapshot));
      practice2LoopScheduler.add(prac2JitterRoutineBegin(snapshot));
      practice2LoopScheduler.add(prac2JitterRoutineEachFrame());
      practice2LoopScheduler.add(prac2JitterRoutineEnd(snapshot));
      practice2LoopScheduler.add(prac2OutRoutineBegin(snapshot));
      practice2LoopScheduler.add(prac2OutRoutineEachFrame());
      practice2LoopScheduler.add(prac2OutRoutineEnd(snapshot));
      practice2LoopScheduler.add(practice2LoopEndIteration(practice2LoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}

async function practice2LoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(practice2);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function practice2LoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}

async function pracBlockLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(pracBlock);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function pracBlockLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}

function runsLoopBegin(runsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    runs = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 1, method: TrialHandler.Method.SEQUENTIAL,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'conditions/Runs.xlsx',
      seed: undefined, name: 'runs'
    });
    psychoJS.experiment.addLoop(runs); // add the loop to the experiment
    currentLoop = runs;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisRun of runs) {
      snapshot = runs.getSnapshot();
      runsLoopScheduler.add(importConditions(snapshot));
      runsLoopScheduler.add(getReadyRoutineBegin(snapshot));
      runsLoopScheduler.add(getReadyRoutineEachFrame());
      runsLoopScheduler.add(getReadyRoutineEnd(snapshot));
      runsLoopScheduler.add(waitForScannerRoutineBegin(snapshot));
      runsLoopScheduler.add(waitForScannerRoutineEachFrame());
      runsLoopScheduler.add(waitForScannerRoutineEnd(snapshot));
      const trialsLoopScheduler = new Scheduler(psychoJS);
      runsLoopScheduler.add(trialsLoopBegin(trialsLoopScheduler, snapshot));
      runsLoopScheduler.add(trialsLoopScheduler);
      runsLoopScheduler.add(trialsLoopEnd);
      runsLoopScheduler.add(breakScreenRoutineBegin(snapshot));
      runsLoopScheduler.add(breakScreenRoutineEachFrame());
      runsLoopScheduler.add(breakScreenRoutineEnd(snapshot));
      runsLoopScheduler.add(runsLoopEndIteration(runsLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}

function trialsLoopBegin(trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 1, method: TrialHandler.Method.SEQUENTIAL,
      extraInfo: expInfo, originPath: undefined,
      trialList: trialOrder,
      seed: undefined, name: 'trials'
    });
    psychoJS.experiment.addLoop(trials); // add the loop to the experiment
    currentLoop = trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisTrial of trials) {
      snapshot = trials.getSnapshot();
      trialsLoopScheduler.add(importConditions(snapshot));
      trialsLoopScheduler.add(cueRoutineBegin(snapshot));
      trialsLoopScheduler.add(cueRoutineEachFrame());
      trialsLoopScheduler.add(cueRoutineEnd(snapshot));
      trialsLoopScheduler.add(outcomeDelayRoutineRoutineBegin(snapshot));
      trialsLoopScheduler.add(outcomeDelayRoutineRoutineEachFrame());
      trialsLoopScheduler.add(outcomeDelayRoutineRoutineEnd(snapshot));
      trialsLoopScheduler.add(outcomeRoutineRoutineBegin(snapshot));
      trialsLoopScheduler.add(outcomeRoutineRoutineEachFrame());
      trialsLoopScheduler.add(outcomeRoutineRoutineEnd(snapshot));
      trialsLoopScheduler.add(fixationRoutineBegin(snapshot));
      trialsLoopScheduler.add(fixationRoutineEachFrame());
      trialsLoopScheduler.add(fixationRoutineEnd(snapshot));
      trialsLoopScheduler.add(trialsLoopEndIteration(trialsLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}

async function trialsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trials);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function trialsLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}

async function runsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(runs);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}

function runsLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}

function pracDir1RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'pracDir1' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    pracDir1Clock.reset();
    routineTimer.reset();
    pracDir1MaxDurationReached = false;
    // update component parameters for each repeat
    // Run 'Begin Routine' code from selectionPracticeCode
    selectionIndicatorPrac.status = NOT_STARTED;
    t_started = false;
    
    pracPress.keys = undefined;
    pracPress.rt = undefined;
    _pracPress_allKeys = [];
    psychoJS.experiment.addData('pracDir1.started', globalClock.getTime());
    pracDir1MaxDuration = null
    // keep track of which components have finished
    pracDir1Components = [];
    pracDir1Components.push(choiceDirText);
    pracDir1Components.push(prac2Image);
    pracDir1Components.push(prac1Image);
    pracDir1Components.push(pracPress);
    
    for (const thisComponent of pracDir1Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}

function pracDir1RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'pracDir1' ---
    // get current time
    t = pracDir1Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *choiceDirText* updates
    if (t >= 0.0 && choiceDirText.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      choiceDirText.tStart = t;  // (not accounting for frame time here)
      choiceDirText.frameNStart = frameN;  // exact frame index
      
      choiceDirText.setAutoDraw(true);
    }
    
    
    // if choiceDirText is active this frame...
    if (choiceDirText.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *prac2Image* updates
    if (t >= 0.0 && prac2Image.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      prac2Image.tStart = t;  // (not accounting for frame time here)
      prac2Image.frameNStart = frameN;  // exact frame index
      
      prac2Image.setAutoDraw(true);
    }
    
    
    // if prac2Image is active this frame...
    if (prac2Image.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *prac1Image* updates
    if (t >= 0.0 && prac1Image.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      prac1Image.tStart = t;  // (not accounting for frame time here)
      prac1Image.frameNStart = frameN;  // exact frame index
      
      prac1Image.setAutoDraw(true);
    }
    
    
    // if prac1Image is active this frame...
    if (prac1Image.status === PsychoJS.Status.STARTED) {
    }
    
    // Run 'Each Frame' code from selectionPracticeCode
    if (t_started) {
        if ((t > (t_started + 5))) {
            continueRoutine = false;
        }
    }
    if ((pracPress.keys && (selectionIndicatorPrac.status === NOT_STARTED))) {
        selectionIndicatorPrac.status = PsychoJS.Status.STARTED;
        selectionIndicatorPrac.setPos(selectionPosition(pracPress.keys));
        selectionIndicatorPrac.setAutoDraw(true);
        t_started = new util.Clock().getTime();
    }
    
    
    // *pracPress* updates
    if (t >= 0.0 && pracPress.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pracPress.tStart = t;  // (not accounting for frame time here)
      pracPress.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { pracPress.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { pracPress.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { pracPress.clearEvents(); });
    }
    
    // if pracPress is active this frame...
    if (pracPress.status === PsychoJS.Status.STARTED) {
      let theseKeys = pracPress.getKeys({keyList: allowedKeys, waitRelease: false});
      _pracPress_allKeys = _pracPress_allKeys.concat(theseKeys);
      if (_pracPress_allKeys.length > 0) {
        pracPress.keys = _pracPress_allKeys[_pracPress_allKeys.length - 1].name;  // just the last key pressed
        pracPress.rt = _pracPress_allKeys[_pracPress_allKeys.length - 1].rt;
        pracPress.duration = _pracPress_allKeys[_pracPress_allKeys.length - 1].duration;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      routineForceEnded = true;
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of pracDir1Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function pracDir1RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'pracDir1' ---
    for (const thisComponent of pracDir1Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('pracDir1.stopped', globalClock.getTime());
    // Run 'End Routine' code from selectionPracticeCode
    selectionIndicatorPrac.setAutoDraw(false);
    
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(pracPress.corr, level);
    }
    psychoJS.experiment.addData('pracPress.keys', pracPress.keys);
    if (typeof pracPress.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('pracPress.rt', pracPress.rt);
        psychoJS.experiment.addData('pracPress.duration', pracPress.duration);
        }
    
    pracPress.stop();
    // the Routine "pracDir1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function pracDir2RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'pracDir2' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    pracDir2Clock.reset();
    routineTimer.reset();
    pracDir2MaxDurationReached = false;
    // update component parameters for each repeat
    key_resp.keys = undefined;
    key_resp.rt = undefined;
    _key_resp_allKeys = [];
    psychoJS.experiment.addData('pracDir2.started', globalClock.getTime());
    pracDir2MaxDuration = null
    // keep track of which components have finished
    pracDir2Components = [];
    pracDir2Components.push(fixDirText);
    pracDir2Components.push(key_resp);
    
    for (const thisComponent of pracDir2Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}

function pracDir2RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'pracDir2' ---
    // get current time
    t = pracDir2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *fixDirText* updates
    if (t >= 0.0 && fixDirText.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      fixDirText.tStart = t;  // (not accounting for frame time here)
      fixDirText.frameNStart = frameN;  // exact frame index
      
      fixDirText.setAutoDraw(true);
    }
    
    
    // if fixDirText is active this frame...
    if (fixDirText.status === PsychoJS.Status.STARTED) {
    }
    
    
    // *key_resp* updates
    if (t >= 0.0 && key_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp.tStart = t;  // (not accounting for frame time here)
      key_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp.clearEvents(); });
    }
    
    // if key_resp is active this frame...
    if (key_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp.getKeys({keyList: 'space', waitRelease: false});
      _key_resp_allKeys = _key_resp_allKeys.concat(theseKeys);
      if (_key_resp_allKeys.length > 0) {
        key_resp.keys = _key_resp_allKeys[_key_resp_allKeys.length - 1].name;  // just the last key pressed
        key_resp.rt = _key_resp_allKeys[_key_resp_allKeys.length - 1].rt;
        key_resp.duration = _key_resp_allKeys[_key_resp_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      routineForceEnded = true;
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of pracDir2Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function pracDir2RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'pracDir2' ---
    for (const thisComponent of pracDir2Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('pracDir2.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp.corr, level);
    }
    psychoJS.experiment.addData('key_resp.keys', key_resp.keys);
    if (typeof key_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp.rt', key_resp.rt);
        psychoJS.experiment.addData('key_resp.duration', key_resp.duration);
        routineTimer.reset();
        }
    
    key_resp.stop();
    // the Routine "pracDir2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}

function pracCueRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'pracCue' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // keep track of whether this Routine was forcibly ended
    routineForceEnded = false;
    pracCueClock.reset(routineTimer.getTime());
    routineTimer.add(2.500000);
    pracCueMaxDurationReached = false;
    // update component parameters for each repeat
    pracLeftImg.setImage(leftImgPath);
    pracRightImg.setImage(rightImgPath);
    pracResp.keys = undefined;
    pracResp.rt = undefined;
    _pracResp_allKeys = [];
    // Run 'Begin Routine' code from selectionCode
    cueEndTime = 2.5;
    selectionIndicatorPrac.status = NOT_STARTED;
    
    psychoJS.experiment.addData('pracCue.started', globalClock.getTime());
    pracCueMaxDuration = null
    // keep track of which components have finished
    pracCueComponents = [];
    pracCueComponents.push(pracLeftImg);
    pracCueComponents.push(pracRightImg);
    pracCueComponents.push(pracResp);
    
    for (const thisComponent of pracCueComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}

function pracCueRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'pracCue' ---
    // get current time
    t = pracCueClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *pracLeftImg* updates
    if (t >= 0.0 && pracLeftImg.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pracLeftImg.tStart = t;  // (not accounting for frame time here)
      pracLeftImg.frameNStart = frameN;  // exact frame index
      
      pracLeftImg.setAutoDraw(true);
    }
    
    
    // if pracLeftImg is active this frame...
    if (pracLeftImg.status === PsychoJS.Status.STARTED) {
    }
    
    frameRemains = 0.0 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (pracLeftImg.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      // keep track of stop time/frame for later
      pracLeftImg.tStop = t;  // not accounting for scr refresh
      pracLeftImg.frameNStop = frameN;  // exact frame index
      // update status
      pracLeftImg.status = PsychoJS.Status.FINISHED;
      pracLeftImg.setAutoDraw(false);
    }
    
    
    // *pracRightImg* updates
    if (t >= 0.0 && pracRightImg.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pracRightImg.tStart = t;  // (not accounting for frame time here)
      pracRightImg.frameNStart = frameN;  // exact frame index
      
      pracRightImg.setAutoDraw(true);
    }
    
    
    // if pracRightImg is active this frame...
    if (pracRightImg.status === PsychoJS.Status.STARTED) {
    }
    
    frameRemains = 0.0 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (pracRightImg.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      // keep track of stop time/frame for later
      pracRightImg.tStop = t;  // not accounting for scr refresh
      pracRightImg.frameNStop = frameN;  // exact frame index
      // update status
      pracRightImg.status = PsychoJS.Status.FINISHED;
      pracRightImg.setAutoDraw(false);
    }
    
    
    // *pracResp* updates
    if (t >= 0.0 && pracResp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pracResp.tStart = t;  // (not accounting for frame time here)
      pracResp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { pracResp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { pracResp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { pracResp.clearEvents(); });
    }
    frameRemains = 0.0 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (pracResp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      // keep track of stop time/frame for later
      pracResp.tStop = t;  // not accounting for scr refresh
      pracResp.frameNStop = frameN;  // exact frame index
      // update status
      pracResp.status = PsychoJS.Status.FINISHED;
      frameRemains = 0.0 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
      if (pracResp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
        // keep track of stop time/frame for later
        pracResp.tStop = t;  // not accounting for scr refresh
        pracResp.frameNStop = frameN;  // exact frame index
        // update status
        pracResp.status = PsychoJS.Status.FINISHED;
        pracResp.status = PsychoJS.Status.FINISHED;
          }
        
      }
      
      // if pracResp is active this frame...
      if (pracResp.status === PsychoJS.Status.STARTED) {
        let theseKeys = pracResp.getKeys({keyList: allowedKeys, waitRelease: false});
        _pracResp_allKeys = _pracResp_allKeys.concat(theseKeys);
        if (_pracResp_allKeys.length > 0) {
          pracResp.keys = _pracResp_allKeys[0].name;  // just the first key pressed
          pracResp.rt = _pracResp_allKeys[0].rt;
          pracResp.duration = _pracResp_allKeys[0].duration;
          // was this correct?
          if (pracResp.keys == PracCorrect) {
              pracResp.corr = 1;
          } else {
              pracResp.corr = 0;
          }
        }
      }
      
      // Run 'Each Frame' code from selectionCode
      if ((t > cueEndTime)) {
          continueRoutine = false;
      }
      if ((pracResp.keys && (selectionIndicatorPrac.status === NOT_STARTED))) {
          selectionIndicatorPrac.status = PsychoJS.Status.STARTED;
          selectionIndicatorPrac.setPos(selectionPosition(pracResp.keys));
          selectionIndicatorPrac.setAutoDraw(true);
      }
      
      // check for quit (typically the Esc key)
      if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
        return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
      }
      
      // check if the Routine should terminate
      if (!continueRoutine) {  // a component has requested a forced-end of Routine
        routineForceEnded = true;
        return Scheduler.Event.NEXT;
      }
      
      continueRoutine = false;  // reverts to True if at least one component still running
      for (const thisComponent of pracCueComponents)
        if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
          continueRoutine = true;
          break;
        }
      
      // refresh the screen if continuing
      if (continueRoutine && routineTimer.getTime() > 0) {
        return Scheduler.Event.FLIP_REPEAT;
      } else {
        return Scheduler.Event.NEXT;
      }
    };
  }
  
  function pracCueRoutineEnd(snapshot) {
    return async function () {
      //--- Ending Routine 'pracCue' ---
      for (const thisComponent of pracCueComponents) {
        if (typeof thisComponent.setAutoDraw === 'function') {
          thisComponent.setAutoDraw(false);
        }
      }
      psychoJS.experiment.addData('pracCue.stopped', globalClock.getTime());
      // was no response the correct answer?!
      if (pracResp.keys === undefined) {
        if (['None','none',undefined].includes(PracCorrect)) {
           pracResp.corr = 1;  // correct non-response
        } else {
           pracResp.corr = 0;  // failed to respond (incorrectly)
        }
      }
      // store data for current loop
      // update the trial handler
      if (currentLoop instanceof MultiStairHandler) {
        currentLoop.addResponse(pracResp.corr, level);
      }
      psychoJS.experiment.addData('pracResp.keys', pracResp.keys);
      psychoJS.experiment.addData('pracResp.corr', pracResp.corr);
      if (typeof pracResp.keys !== 'undefined') {  // we had a response
          psychoJS.experiment.addData('pracResp.rt', pracResp.rt);
          psychoJS.experiment.addData('pracResp.duration', pracResp.duration);
          }
      
      pracResp.stop();
      // Run 'End Routine' code from selectionCode
      selectionIndicatorPrac.setAutoDraw(false);
      console.log((pracResp.corr === 1));
      
      if (routineForceEnded) {
          routineTimer.reset();} else if (pracCueMaxDurationReached) {
          pracCueClock.add(pracCueMaxDuration);
      } else {
          pracCueClock.add(2.500000);
      }
      // Routines running outside a loop should always advance the datafile row
      if (currentLoop === psychoJS.experiment) {
        psychoJS.experiment.nextEntry(snapshot);
      }
      return Scheduler.Event.NEXT;
    }
  }
  
  function pracJitterRoutineBegin(snapshot) {
    return async function () {
      TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
      
      //--- Prepare to start Routine 'pracJitter' ---
      t = 0;
      frameN = -1;
      continueRoutine = true; // until we're told otherwise
      // keep track of whether this Routine was forcibly ended
      routineForceEnded = false;
      pracJitterClock.reset(routineTimer.getTime());
      routineTimer.add(2.500000);
      pracJitterMaxDurationReached = false;
      // update component parameters for each repeat
      // Run 'Begin Routine' code from outcomeCode
      if (pracResp.keys) {
          if ((pracResp.keys === PracCorrect[practice.thisN])) {
              pracOutcome = "Correct!";
          } else {
              pracOutcome = "Incorrect!";
          }
      } else {
          pracOutcome = "Missed";
      }
      
      psychoJS.experiment.addData('pracJitter.started', globalClock.getTime());
      pracJitterMaxDuration = null
      // keep track of which components have finished
      pracJitterComponents = [];
      pracJitterComponents.push(pracFix);
      
      for (const thisComponent of pracJitterComponents)
        if ('status' in thisComponent)
          thisComponent.status = PsychoJS.Status.NOT_STARTED;
      return Scheduler.Event.NEXT;
    }
  }
  
  function pracJitterRoutineEachFrame() {
    return async function () {
      //--- Loop for each frame of Routine 'pracJitter' ---
      // get current time
      t = pracJitterClock.getTime();
      frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
      // update/draw components on each frame
      
      // *pracFix* updates
      if (t >= 0.0 && pracFix.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        pracFix.tStart = t;  // (not accounting for frame time here)
        pracFix.frameNStart = frameN;  // exact frame index
        
        pracFix.setAutoDraw(true);
      }
      
      
      // if pracFix is active this frame...
      if (pracFix.status === PsychoJS.Status.STARTED) {
      }
      
      frameRemains = 0.0 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
      if (pracFix.status === PsychoJS.Status.STARTED && t >= frameRemains) {
        // keep track of stop time/frame for later
        pracFix.tStop = t;  // not accounting for scr refresh
        pracFix.frameNStop = frameN;  // exact frame index
        // update status
        pracFix.status = PsychoJS.Status.FINISHED;
        pracFix.setAutoDraw(false);
      }
      
      // check for quit (typically the Esc key)
      if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
        return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
      }
      
      // check if the Routine should terminate
      if (!continueRoutine) {  // a component has requested a forced-end of Routine
        routineForceEnded = true;
        return Scheduler.Event.NEXT;
      }
      
      continueRoutine = false;  // reverts to True if at least one component still running
      for (const thisComponent of pracJitterComponents)
        if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
          continueRoutine = true;
          break;
        }
      
      // refresh the screen if continuing
      if (continueRoutine && routineTimer.getTime() > 0) {
        return Scheduler.Event.FLIP_REPEAT;
      } else {
        return Scheduler.Event.NEXT;
      }
    };
  }
  
  function pracJitterRoutineEnd(snapshot) {
    return async function () {
      //--- Ending Routine 'pracJitter' ---
      for (const thisComponent of pracJitterComponents) {
        if (typeof thisComponent.setAutoDraw === 'function') {
          thisComponent.setAutoDraw(false);
        }
      }
      psychoJS.experiment.addData('pracJitter.stopped', globalClock.getTime());
      if (routineForceEnded) {
          routineTimer.reset();} else if (pracJitterMaxDurationReached) {
          pracJitterClock.add(pracJitterMaxDuration);
      } else {
          pracJitterClock.add(2.500000);
      }
      // Routines running outside a loop should always advance the datafile row
      if (currentLoop === psychoJS.experiment) {
        psychoJS.experiment.nextEntry(snapshot);
      }
      return Scheduler.Event.NEXT;
    }
  }
  
  function pracOutRoutineBegin(snapshot) {
    return async function () {
      TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
      
      //--- Prepare to start Routine 'pracOut' ---
      t = 0;
      frameN = -1;
      continueRoutine = true; // until we're told otherwise
      // keep track of whether this Routine was forcibly ended
      routineForceEnded = false;
      pracOutClock.reset(routineTimer.getTime());
      routineTimer.add(2.500000);
      pracOutMaxDurationReached = false;
      // update component parameters for each repeat
      pracOutText.setText(pracOutcome);
      psychoJS.experiment.addData('pracOut.started', globalClock.getTime());
      pracOutMaxDuration = null
      // keep track of which components have finished
      pracOutComponents = [];
      pracOutComponents.push(pracOutText);
      pracOutComponents.push(pracOutFix);
      
      for (const thisComponent of pracOutComponents)
        if ('status' in thisComponent)
          thisComponent.status = PsychoJS.Status.NOT_STARTED;
      return Scheduler.Event.NEXT;
    }
  }
  
  function pracOutRoutineEachFrame() {
    return async function () {
      //--- Loop for each frame of Routine 'pracOut' ---
      // get current time
      t = pracOutClock.getTime();
      frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
      // update/draw components on each frame
      
      // *pracOutText* updates
      if (t >= 0.0 && pracOutText.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        pracOutText.tStart = t;  // (not accounting for frame time here)
        pracOutText.frameNStart = frameN;  // exact frame index
        
        pracOutText.setAutoDraw(true);
      }
      
      
      // if pracOutText is active this frame...
      if (pracOutText.status === PsychoJS.Status.STARTED) {
      }
      
      frameRemains = 0.0 + 1.0 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
      if (pracOutText.status === PsychoJS.Status.STARTED && t >= frameRemains) {
        // keep track of stop time/frame for later
        pracOutText.tStop = t;  // not accounting for scr refresh
        pracOutText.frameNStop = frameN;  // exact frame index
        // update status
        pracOutText.status = PsychoJS.Status.FINISHED;
        pracOutText.setAutoDraw(false);
      }
      
      
      // *pracOutFix* updates
      if (t >= 1.0 && pracOutFix.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        pracOutFix.tStart = t;  // (not accounting for frame time here)
        pracOutFix.frameNStart = frameN;  // exact frame index
        
        pracOutFix.setAutoDraw(true);
      }
      
      
      // if pracOutFix is active this frame...
      if (pracOutFix.status === PsychoJS.Status.STARTED) {
      }
      
      frameRemains = 1.0 + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
      if (pracOutFix.status === PsychoJS.Status.STARTED && t >= frameRemains) {
        // keep track of stop time/frame for later
        pracOutFix.tStop = t;  // not accounting for scr refresh
        pracOutFix.frameNStop = frameN;  // exact frame index
        // update status
        pracOutFix.status = PsychoJS.Status.FINISHED;
        pracOutFix.setAutoDraw(false);
      }
      
      // check for quit (typically the Esc key)
      if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
        return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
      }
      
      // check if the Routine should terminate
      if (!continueRoutine) {  // a component has requested a forced-end of Routine
        routineForceEnded = true;
        return Scheduler.Event.NEXT;
      }
      
      continueRoutine = false;  // reverts to True if at least one component still running
      for (const thisComponent of pracOutComponents)
        if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
          continueRoutine = true;
          break;
        }
      
      // refresh the screen if continuing
      if (continueRoutine && routineTimer.getTime() > 0) {
        return Scheduler.Event.FLIP_REPEAT;
      } else {
        return Scheduler.Event.NEXT;
      }
    };
  }
  
  function pracOutRoutineEnd(snapshot) {
    return async function () {
      //--- Ending Routine 'pracOut' ---
      for (const thisComponent of pracOutComponents) {
        if (typeof thisComponent.setAutoDraw === 'function') {
          thisComponent.setAutoDraw(false);
        }
      }
      psychoJS.experiment.addData('pracOut.stopped', globalClock.getTime());
      if (routineForceEnded) {
          routineTimer.reset();} else if (pracOutMaxDurationReached) {
          pracOutClock.add(pracOutMaxDuration);
      } else {
          pracOutClock.add(2.500000);
      }
      // Routines running outside a loop should always advance the datafile row
      if (currentLoop === psychoJS.experiment) {
        psychoJS.experiment.nextEntry(snapshot);
      }
      return Scheduler.Event.NEXT;
    }
  }
  
  function check1RoutineBegin(snapshot) {
    return async function () {
      TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
      
      //--- Prepare to start Routine 'check1' ---
      t = 0;
      frameN = -1;
      continueRoutine = true; // until we're told otherwise
      // keep track of whether this Routine was forcibly ended
      routineForceEnded = false;
      check1Clock.reset();
      routineTimer.reset();
      check1MaxDurationReached = false;
      // update component parameters for each repeat
      // Run 'Begin Routine' code from check1Select
      selectionIndicatorPrac.status = NOT_STARTED;
      t_started = false;
      
      check1Press.keys = undefined;
      check1Press.rt = undefined;
      _check1Press_allKeys = [];
      down1.keys = undefined;
      down1.rt = undefined;
      _down1_allKeys = [];
      psychoJS.experiment.addData('check1.started', globalClock.getTime());
      check1MaxDuration = null
      // keep track of which components have finished
      check1Components = [];
      check1Components.push(check1Text);
      check1Components.push(check1ImageLeft);
      check1Components.push(check1ImageRight);
      check1Components.push(check1Press);
      check1Components.push(down1);
      
      for (const thisComponent of check1Components)
        if ('status' in thisComponent)
          thisComponent.status = PsychoJS.Status.NOT_STARTED;
      return Scheduler.Event.NEXT;
    }
  }
  
  function check1RoutineEachFrame() {
    return async function () {
      //--- Loop for each frame of Routine 'check1' ---
      // get current time
      t = check1Clock.getTime();
      frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
      // update/draw components on each frame
      
      // *check1Text* updates
      if (t >= 0.0 && check1Text.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        check1Text.tStart = t;  // (not accounting for frame time here)
        check1Text.frameNStart = frameN;  // exact frame index
        
        check1Text.setAutoDraw(true);
      }
      
      
      // if check1Text is active this frame...
      if (check1Text.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *check1ImageLeft* updates
      if (t >= 0.0 && check1ImageLeft.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        check1ImageLeft.tStart = t;  // (not accounting for frame time here)
        check1ImageLeft.frameNStart = frameN;  // exact frame index
        
        check1ImageLeft.setAutoDraw(true);
      }
      
      
      // if check1ImageLeft is active this frame...
      if (check1ImageLeft.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *check1ImageRight* updates
      if (t >= 0.0 && check1ImageRight.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        check1ImageRight.tStart = t;  // (not accounting for frame time here)
        check1ImageRight.frameNStart = frameN;  // exact frame index
        
        check1ImageRight.setAutoDraw(true);
      }
      
      
      // if check1ImageRight is active this frame...
      if (check1ImageRight.status === PsychoJS.Status.STARTED) {
      }
      
      // Run 'Each Frame' code from check1Select
      if (t_started) {
          if ((t > (t_started + 5))) {
              continueRoutine = false;
          }
      }
      if ((check1Press.keys && (selectionIndicatorPrac.status === NOT_STARTED))) {
          selectionIndicatorPrac.status = PsychoJS.Status.STARTED;
          selectionIndicatorPrac.setPos(selectionPosition(check1Press.keys));
          selectionIndicatorPrac.setAutoDraw(true);
          t_started = new util.Clock().getTime();
      }
      
      
      // *check1Press* updates
      if (t >= 0.0 && check1Press.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        check1Press.tStart = t;  // (not accounting for frame time here)
        check1Press.frameNStart = frameN;  // exact frame index
        
        // keyboard checking is just starting
        psychoJS.window.callOnFlip(function() { check1Press.clock.reset(); });  // t=0 on next screen flip
        psychoJS.window.callOnFlip(function() { check1Press.start(); }); // start on screen flip
        psychoJS.window.callOnFlip(function() { check1Press.clearEvents(); });
      }
      
      // if check1Press is active this frame...
      if (check1Press.status === PsychoJS.Status.STARTED) {
        let theseKeys = check1Press.getKeys({keyList: allowedKeys, waitRelease: false});
        _check1Press_allKeys = _check1Press_allKeys.concat(theseKeys);
        if (_check1Press_allKeys.length > 0) {
          check1Press.keys = _check1Press_allKeys[_check1Press_allKeys.length - 1].name;  // just the last key pressed
          check1Press.rt = _check1Press_allKeys[_check1Press_allKeys.length - 1].rt;
          check1Press.duration = _check1Press_allKeys[_check1Press_allKeys.length - 1].duration;
        }
      }
      
      
      // *down1* updates
      if (t >= 0.0 && down1.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        down1.tStart = t;  // (not accounting for frame time here)
        down1.frameNStart = frameN;  // exact frame index
        
        // keyboard checking is just starting
        psychoJS.window.callOnFlip(function() { down1.clock.reset(); });  // t=0 on next screen flip
        psychoJS.window.callOnFlip(function() { down1.start(); }); // start on screen flip
        psychoJS.window.callOnFlip(function() { down1.clearEvents(); });
      }
      
      // if down1 is active this frame...
      if (down1.status === PsychoJS.Status.STARTED) {
        let theseKeys = down1.getKeys({keyList: 'down', waitRelease: false});
        _down1_allKeys = _down1_allKeys.concat(theseKeys);
        if (_down1_allKeys.length > 0) {
          down1.keys = _down1_allKeys[_down1_allKeys.length - 1].name;  // just the last key pressed
          down1.rt = _down1_allKeys[_down1_allKeys.length - 1].rt;
          down1.duration = _down1_allKeys[_down1_allKeys.length - 1].duration;
          // a response ends the routine
          continueRoutine = false;
        }
      }
      
      // check for quit (typically the Esc key)
      if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
        return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
      }
      
      // check if the Routine should terminate
      if (!continueRoutine) {  // a component has requested a forced-end of Routine
        routineForceEnded = true;
        return Scheduler.Event.NEXT;
      }
      
      continueRoutine = false;  // reverts to True if at least one component still running
      for (const thisComponent of check1Components)
        if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
          continueRoutine = true;
          break;
        }
      
      // refresh the screen if continuing
      if (continueRoutine) {
        return Scheduler.Event.FLIP_REPEAT;
      } else {
        return Scheduler.Event.NEXT;
      }
    };
  }
  
  function check1RoutineEnd(snapshot) {
    return async function () {
      //--- Ending Routine 'check1' ---
      for (const thisComponent of check1Components) {
        if (typeof thisComponent.setAutoDraw === 'function') {
          thisComponent.setAutoDraw(false);
        }
      }
      psychoJS.experiment.addData('check1.stopped', globalClock.getTime());
      // Run 'End Routine' code from check1Select
      selectionIndicatorPrac.setAutoDraw(false);
      if ((check1Press.keys === "right")) {
          currentLoop.addData("check1", "correct");
      } else {
          currentLoop.addData("check1", "incorrect");
      }
      
      // update the trial handler
      if (currentLoop instanceof MultiStairHandler) {
        currentLoop.addResponse(check1Press.corr, level);
      }
      psychoJS.experiment.addData('check1Press.keys', check1Press.keys);
      if (typeof check1Press.keys !== 'undefined') {  // we had a response
          psychoJS.experiment.addData('check1Press.rt', check1Press.rt);
          psychoJS.experiment.addData('check1Press.duration', check1Press.duration);
          }
      
      check1Press.stop();
      // update the trial handler
      if (currentLoop instanceof MultiStairHandler) {
        currentLoop.addResponse(down1.corr, level);
      }
      psychoJS.experiment.addData('down1.keys', down1.keys);
      if (typeof down1.keys !== 'undefined') {  // we had a response
          psychoJS.experiment.addData('down1.rt', down1.rt);
          psychoJS.experiment.addData('down1.duration', down1.duration);
          routineTimer.reset();
          }
      
      down1.stop();
      // the Routine "check1" was not non-slip safe, so reset the non-slip timer
      routineTimer.reset();
      
      // Routines running outside a loop should always advance the datafile row
      if (currentLoop === psychoJS.experiment) {
        psychoJS.experiment.nextEntry(snapshot);
      }
      return Scheduler.Event.NEXT;
    }
  }
  
  function phase2DirRoutineBegin(snapshot) {
    return async function () {
      TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
      
      //--- Prepare to start Routine 'phase2Dir' ---
      t = 0;
      frameN = -1;
      continueRoutine = true; // until we're told otherwise
      // keep track of whether this Routine was forcibly ended
      routineForceEnded = false;
      phase2DirClock.reset();
      routineTimer.reset();
      phase2DirMaxDurationReached = false;
      // update component parameters for each repeat
      phase2Press.keys = undefined;
      phase2Press.rt = undefined;
      _phase2Press_allKeys = [];
      psychoJS.experiment.addData('phase2Dir.started', globalClock.getTime());
      phase2DirMaxDuration = null
      // keep track of which components have finished
      phase2DirComponents = [];
      phase2DirComponents.push(phase2Instruc);
      phase2DirComponents.push(phase2Press);
      
      for (const thisComponent of phase2DirComponents)
        if ('status' in thisComponent)
          thisComponent.status = PsychoJS.Status.NOT_STARTED;
      return Scheduler.Event.NEXT;
    }
  }
  
  function phase2DirRoutineEachFrame() {
    return async function () {
      //--- Loop for each frame of Routine 'phase2Dir' ---
      // get current time
      t = phase2DirClock.getTime();
      frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
      // update/draw components on each frame
      
      // *phase2Instruc* updates
      if (t >= 0.0 && phase2Instruc.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        phase2Instruc.tStart = t;  // (not accounting for frame time here)
        phase2Instruc.frameNStart = frameN;  // exact frame index
        
        phase2Instruc.setAutoDraw(true);
      }
      
      
      // if phase2Instruc is active this frame...
      if (phase2Instruc.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *phase2Press* updates
      if (t >= 0.0 && phase2Press.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        phase2Press.tStart = t;  // (not accounting for frame time here)
        phase2Press.frameNStart = frameN;  // exact frame index
        
        // keyboard checking is just starting
        psychoJS.window.callOnFlip(function() { phase2Press.clock.reset(); });  // t=0 on next screen flip
        psychoJS.window.callOnFlip(function() { phase2Press.start(); }); // start on screen flip
        psychoJS.window.callOnFlip(function() { phase2Press.clearEvents(); });
      }
      
      // if phase2Press is active this frame...
      if (phase2Press.status === PsychoJS.Status.STARTED) {
        let theseKeys = phase2Press.getKeys({keyList: 'space', waitRelease: false});
        _phase2Press_allKeys = _phase2Press_allKeys.concat(theseKeys);
        if (_phase2Press_allKeys.length > 0) {
          phase2Press.keys = _phase2Press_allKeys[_phase2Press_allKeys.length - 1].name;  // just the last key pressed
          phase2Press.rt = _phase2Press_allKeys[_phase2Press_allKeys.length - 1].rt;
          phase2Press.duration = _phase2Press_allKeys[_phase2Press_allKeys.length - 1].duration;
          // a response ends the routine
          continueRoutine = false;
        }
      }
      
      // check for quit (typically the Esc key)
      if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
        return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
      }
      
      // check if the Routine should terminate
      if (!continueRoutine) {  // a component has requested a forced-end of Routine
        routineForceEnded = true;
        return Scheduler.Event.NEXT;
      }
      
      continueRoutine = false;  // reverts to True if at least one component still running
      for (const thisComponent of phase2DirComponents)
        if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
          continueRoutine = true;
          break;
        }
      
      // refresh the screen if continuing
      if (continueRoutine) {
        return Scheduler.Event.FLIP_REPEAT;
      } else {
        return Scheduler.Event.NEXT;
      }
    };
  }
  
  function phase2DirRoutineEnd(snapshot) {
    return async function () {
      //--- Ending Routine 'phase2Dir' ---
      for (const thisComponent of phase2DirComponents) {
        if (typeof thisComponent.setAutoDraw === 'function') {
          thisComponent.setAutoDraw(false);
        }
      }
      psychoJS.experiment.addData('phase2Dir.stopped', globalClock.getTime());
      // update the trial handler
      if (currentLoop instanceof MultiStairHandler) {
        currentLoop.addResponse(phase2Press.corr, level);
      }
      psychoJS.experiment.addData('phase2Press.keys', phase2Press.keys);
      if (typeof phase2Press.keys !== 'undefined') {  // we had a response
          psychoJS.experiment.addData('phase2Press.rt', phase2Press.rt);
          psychoJS.experiment.addData('phase2Press.duration', phase2Press.duration);
          routineTimer.reset();
          }
      
      phase2Press.stop();
      // the Routine "phase2Dir" was not non-slip safe, so reset the non-slip timer
      routineTimer.reset();
      
      // Routines running outside a loop should always advance the datafile row
      if (currentLoop === psychoJS.experiment) {
        psychoJS.experiment.nextEntry(snapshot);
      }
      return Scheduler.Event.NEXT;
    }
  }
  
  function gainDirsRoutineBegin(snapshot) {
    return async function () {
      TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
      
      //--- Prepare to start Routine 'gainDirs' ---
      t = 0;
      frameN = -1;
      continueRoutine = true; // until we're told otherwise
      // keep track of whether this Routine was forcibly ended
      routineForceEnded = false;
      gainDirsClock.reset();
      routineTimer.reset();
      gainDirsMaxDurationReached = false;
      // update component parameters for each repeat
      advanceScreenPress3.keys = undefined;
      advanceScreenPress3.rt = undefined;
      _advanceScreenPress3_allKeys = [];
      highgainframe.setLineColor(new util.Color('white'));
      lowgainframe.setLineColor(new util.Color('white'));
      // Run 'Begin Routine' code from gainDirText_2
      gainDirText.alignText = "left";
      
      psychoJS.experiment.addData('gainDirs.started', globalClock.getTime());
      gainDirsMaxDuration = null
      // keep track of which components have finished
      gainDirsComponents = [];
      gainDirsComponents.push(advanceScreenPress3);
      gainDirsComponents.push(gainDirText);
      gainDirsComponents.push(highgainframe);
      gainDirsComponents.push(lowgainframe);
      gainDirsComponents.push(highgainTopLabel);
      gainDirsComponents.push(lowgainTopLabel);
      gainDirsComponents.push(highgainBottomLabel);
      gainDirsComponents.push(lowgainBottomLabel);
      gainDirsComponents.push(highGainLabel);
      gainDirsComponents.push(lowGainLabel);
      
      for (const thisComponent of gainDirsComponents)
        if ('status' in thisComponent)
          thisComponent.status = PsychoJS.Status.NOT_STARTED;
      return Scheduler.Event.NEXT;
    }
  }
  
  function gainDirsRoutineEachFrame() {
    return async function () {
      //--- Loop for each frame of Routine 'gainDirs' ---
      // get current time
      t = gainDirsClock.getTime();
      frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
      // update/draw components on each frame
      
      // *advanceScreenPress3* updates
      if (t >= 0.0 && advanceScreenPress3.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        advanceScreenPress3.tStart = t;  // (not accounting for frame time here)
        advanceScreenPress3.frameNStart = frameN;  // exact frame index
        
        // keyboard checking is just starting
        psychoJS.window.callOnFlip(function() { advanceScreenPress3.clock.reset(); });  // t=0 on next screen flip
        psychoJS.window.callOnFlip(function() { advanceScreenPress3.start(); }); // start on screen flip
        psychoJS.window.callOnFlip(function() { advanceScreenPress3.clearEvents(); });
      }
      
      // if advanceScreenPress3 is active this frame...
      if (advanceScreenPress3.status === PsychoJS.Status.STARTED) {
        let theseKeys = advanceScreenPress3.getKeys({keyList: 'space', waitRelease: false});
        _advanceScreenPress3_allKeys = _advanceScreenPress3_allKeys.concat(theseKeys);
        if (_advanceScreenPress3_allKeys.length > 0) {
          advanceScreenPress3.keys = _advanceScreenPress3_allKeys[_advanceScreenPress3_allKeys.length - 1].name;  // just the last key pressed
          advanceScreenPress3.rt = _advanceScreenPress3_allKeys[_advanceScreenPress3_allKeys.length - 1].rt;
          advanceScreenPress3.duration = _advanceScreenPress3_allKeys[_advanceScreenPress3_allKeys.length - 1].duration;
          // a response ends the routine
          continueRoutine = false;
        }
      }
      
      
      // *gainDirText* updates
      if (t >= 0.0 && gainDirText.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        gainDirText.tStart = t;  // (not accounting for frame time here)
        gainDirText.frameNStart = frameN;  // exact frame index
        
        gainDirText.setAutoDraw(true);
      }
      
      
      // if gainDirText is active this frame...
      if (gainDirText.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *highgainframe* updates
      if (t >= 0.0 && highgainframe.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        highgainframe.tStart = t;  // (not accounting for frame time here)
        highgainframe.frameNStart = frameN;  // exact frame index
        
        highgainframe.setAutoDraw(true);
      }
      
      
      // if highgainframe is active this frame...
      if (highgainframe.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *lowgainframe* updates
      if (t >= 0.0 && lowgainframe.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        lowgainframe.tStart = t;  // (not accounting for frame time here)
        lowgainframe.frameNStart = frameN;  // exact frame index
        
        lowgainframe.setAutoDraw(true);
      }
      
      
      // if lowgainframe is active this frame...
      if (lowgainframe.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *highgainTopLabel* updates
      if (t >= 0.0 && highgainTopLabel.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        highgainTopLabel.tStart = t;  // (not accounting for frame time here)
        highgainTopLabel.frameNStart = frameN;  // exact frame index
        
        highgainTopLabel.setAutoDraw(true);
      }
      
      
      // if highgainTopLabel is active this frame...
      if (highgainTopLabel.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *lowgainTopLabel* updates
      if (t >= 0.0 && lowgainTopLabel.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        lowgainTopLabel.tStart = t;  // (not accounting for frame time here)
        lowgainTopLabel.frameNStart = frameN;  // exact frame index
        
        lowgainTopLabel.setAutoDraw(true);
      }
      
      
      // if lowgainTopLabel is active this frame...
      if (lowgainTopLabel.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *highgainBottomLabel* updates
      if (t >= 0.0 && highgainBottomLabel.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        highgainBottomLabel.tStart = t;  // (not accounting for frame time here)
        highgainBottomLabel.frameNStart = frameN;  // exact frame index
        
        highgainBottomLabel.setAutoDraw(true);
      }
      
      
      // if highgainBottomLabel is active this frame...
      if (highgainBottomLabel.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *lowgainBottomLabel* updates
      if (t >= 0.0 && lowgainBottomLabel.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        lowgainBottomLabel.tStart = t;  // (not accounting for frame time here)
        lowgainBottomLabel.frameNStart = frameN;  // exact frame index
        
        lowgainBottomLabel.setAutoDraw(true);
      }
      
      
      // if lowgainBottomLabel is active this frame...
      if (lowgainBottomLabel.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *highGainLabel* updates
      if (t >= 0.0 && highGainLabel.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        highGainLabel.tStart = t;  // (not accounting for frame time here)
        highGainLabel.frameNStart = frameN;  // exact frame index
        
        highGainLabel.setAutoDraw(true);
      }
      
      
      // if highGainLabel is active this frame...
      if (highGainLabel.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *lowGainLabel* updates
      if (t >= 0.0 && lowGainLabel.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        lowGainLabel.tStart = t;  // (not accounting for frame time here)
        lowGainLabel.frameNStart = frameN;  // exact frame index
        
        lowGainLabel.setAutoDraw(true);
      }
      
      
      // if lowGainLabel is active this frame...
      if (lowGainLabel.status === PsychoJS.Status.STARTED) {
      }
      
      // check for quit (typically the Esc key)
      if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
        return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
      }
      
      // check if the Routine should terminate
      if (!continueRoutine) {  // a component has requested a forced-end of Routine
        routineForceEnded = true;
        return Scheduler.Event.NEXT;
      }
      
      continueRoutine = false;  // reverts to True if at least one component still running
      for (const thisComponent of gainDirsComponents)
        if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
          continueRoutine = true;
          break;
        }
      
      // refresh the screen if continuing
      if (continueRoutine) {
        return Scheduler.Event.FLIP_REPEAT;
      } else {
        return Scheduler.Event.NEXT;
      }
    };
  }
  
  function gainDirsRoutineEnd(snapshot) {
    return async function () {
      //--- Ending Routine 'gainDirs' ---
      for (const thisComponent of gainDirsComponents) {
        if (typeof thisComponent.setAutoDraw === 'function') {
          thisComponent.setAutoDraw(false);
        }
      }
      psychoJS.experiment.addData('gainDirs.stopped', globalClock.getTime());
      // update the trial handler
      if (currentLoop instanceof MultiStairHandler) {
        currentLoop.addResponse(advanceScreenPress3.corr, level);
      }
      psychoJS.experiment.addData('advanceScreenPress3.keys', advanceScreenPress3.keys);
      if (typeof advanceScreenPress3.keys !== 'undefined') {  // we had a response
          psychoJS.experiment.addData('advanceScreenPress3.rt', advanceScreenPress3.rt);
          psychoJS.experiment.addData('advanceScreenPress3.duration', advanceScreenPress3.duration);
          routineTimer.reset();
          }
      
      advanceScreenPress3.stop();
      // the Routine "gainDirs" was not non-slip safe, so reset the non-slip timer
      routineTimer.reset();
      
      // Routines running outside a loop should always advance the datafile row
      if (currentLoop === psychoJS.experiment) {
        psychoJS.experiment.nextEntry(snapshot);
      }
      return Scheduler.Event.NEXT;
    }
  }
  
  function lossDirsRoutineBegin(snapshot) {
    return async function () {
      TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
      
      //--- Prepare to start Routine 'lossDirs' ---
      t = 0;
      frameN = -1;
      continueRoutine = true; // until we're told otherwise
      // keep track of whether this Routine was forcibly ended
      routineForceEnded = false;
      lossDirsClock.reset();
      routineTimer.reset();
      lossDirsMaxDurationReached = false;
      // update component parameters for each repeat
      advanceScreen4.keys = undefined;
      advanceScreen4.rt = undefined;
      _advanceScreen4_allKeys = [];
      highLossFrame.setLineColor(new util.Color('white'));
      lowLossFrame.setLineColor(new util.Color('white'));
      // Run 'Begin Routine' code from lossTextCode
      lossDirText.alignText = "left";
      
      psychoJS.experiment.addData('lossDirs.started', globalClock.getTime());
      lossDirsMaxDuration = null
      // keep track of which components have finished
      lossDirsComponents = [];
      lossDirsComponents.push(advanceScreen4);
      lossDirsComponents.push(lossDirText);
      lossDirsComponents.push(highLossFrame);
      lossDirsComponents.push(lowLossFrame);
      lossDirsComponents.push(highLossTopLabel);
      lossDirsComponents.push(lowLossTopLabel);
      lossDirsComponents.push(highLossBottomLabel);
      lossDirsComponents.push(lowLossBottomLabel);
      lossDirsComponents.push(highLossLabel);
      lossDirsComponents.push(lowLossLabel);
      
      for (const thisComponent of lossDirsComponents)
        if ('status' in thisComponent)
          thisComponent.status = PsychoJS.Status.NOT_STARTED;
      return Scheduler.Event.NEXT;
    }
  }
  
  function lossDirsRoutineEachFrame() {
    return async function () {
      //--- Loop for each frame of Routine 'lossDirs' ---
      // get current time
      t = lossDirsClock.getTime();
      frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
      // update/draw components on each frame
      
      // *advanceScreen4* updates
      if (t >= 0.0 && advanceScreen4.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        advanceScreen4.tStart = t;  // (not accounting for frame time here)
        advanceScreen4.frameNStart = frameN;  // exact frame index
        
        // keyboard checking is just starting
        psychoJS.window.callOnFlip(function() { advanceScreen4.clock.reset(); });  // t=0 on next screen flip
        psychoJS.window.callOnFlip(function() { advanceScreen4.start(); }); // start on screen flip
        psychoJS.window.callOnFlip(function() { advanceScreen4.clearEvents(); });
      }
      
      // if advanceScreen4 is active this frame...
      if (advanceScreen4.status === PsychoJS.Status.STARTED) {
        let theseKeys = advanceScreen4.getKeys({keyList: 'space', waitRelease: false});
        _advanceScreen4_allKeys = _advanceScreen4_allKeys.concat(theseKeys);
        if (_advanceScreen4_allKeys.length > 0) {
          advanceScreen4.keys = _advanceScreen4_allKeys[_advanceScreen4_allKeys.length - 1].name;  // just the last key pressed
          advanceScreen4.rt = _advanceScreen4_allKeys[_advanceScreen4_allKeys.length - 1].rt;
          advanceScreen4.duration = _advanceScreen4_allKeys[_advanceScreen4_allKeys.length - 1].duration;
          // a response ends the routine
          continueRoutine = false;
        }
      }
      
      
      // *lossDirText* updates
      if (t >= 0.0 && lossDirText.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        lossDirText.tStart = t;  // (not accounting for frame time here)
        lossDirText.frameNStart = frameN;  // exact frame index
        
        lossDirText.setAutoDraw(true);
      }
      
      
      // if lossDirText is active this frame...
      if (lossDirText.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *highLossFrame* updates
      if (t >= 0.0 && highLossFrame.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        highLossFrame.tStart = t;  // (not accounting for frame time here)
        highLossFrame.frameNStart = frameN;  // exact frame index
        
        highLossFrame.setAutoDraw(true);
      }
      
      
      // if highLossFrame is active this frame...
      if (highLossFrame.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *lowLossFrame* updates
      if (t >= 0.0 && lowLossFrame.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        lowLossFrame.tStart = t;  // (not accounting for frame time here)
        lowLossFrame.frameNStart = frameN;  // exact frame index
        
        lowLossFrame.setAutoDraw(true);
      }
      
      
      // if lowLossFrame is active this frame...
      if (lowLossFrame.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *highLossTopLabel* updates
      if (t >= 0.0 && highLossTopLabel.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        highLossTopLabel.tStart = t;  // (not accounting for frame time here)
        highLossTopLabel.frameNStart = frameN;  // exact frame index
        
        highLossTopLabel.setAutoDraw(true);
      }
      
      
      // if highLossTopLabel is active this frame...
      if (highLossTopLabel.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *lowLossTopLabel* updates
      if (t >= 0.0 && lowLossTopLabel.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        lowLossTopLabel.tStart = t;  // (not accounting for frame time here)
        lowLossTopLabel.frameNStart = frameN;  // exact frame index
        
        lowLossTopLabel.setAutoDraw(true);
      }
      
      
      // if lowLossTopLabel is active this frame...
      if (lowLossTopLabel.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *highLossBottomLabel* updates
      if (t >= 0.0 && highLossBottomLabel.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        highLossBottomLabel.tStart = t;  // (not accounting for frame time here)
        highLossBottomLabel.frameNStart = frameN;  // exact frame index
        
        highLossBottomLabel.setAutoDraw(true);
      }
      
      
      // if highLossBottomLabel is active this frame...
      if (highLossBottomLabel.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *lowLossBottomLabel* updates
      if (t >= 0.0 && lowLossBottomLabel.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        lowLossBottomLabel.tStart = t;  // (not accounting for frame time here)
        lowLossBottomLabel.frameNStart = frameN;  // exact frame index
        
        lowLossBottomLabel.setAutoDraw(true);
      }
      
      
      // if lowLossBottomLabel is active this frame...
      if (lowLossBottomLabel.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *highLossLabel* updates
      if (t >= 0.0 && highLossLabel.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        highLossLabel.tStart = t;  // (not accounting for frame time here)
        highLossLabel.frameNStart = frameN;  // exact frame index
        
        highLossLabel.setAutoDraw(true);
      }
      
      
      // if highLossLabel is active this frame...
      if (highLossLabel.status === PsychoJS.Status.STARTED) {
      }
      
      
      // *lowLossLabel* updates
      if (t >= 0.0 && lowLossLabel.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        lowLossLabel.tStart = t;  // (not accounting for frame time here)
        lowLossLabel.frameNStart = frameN;  // exact frame index
        
        lowLossLabel.setAutoDraw(true);
      }
      
      
      // if lowLossLabel is active this frame...
      if (lowLossLabel.status === PsychoJS.Status.STARTED) {
      }
      
      // check for quit (typically the Esc key)
      if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
        return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
      }
      
      // check if the Routine should terminate
      if (!continueRoutine) {  // a component has requested a forced-end of Routine
        routineForceEnded = true;
        return Scheduler.Event.NEXT;
      }
      
      continueRoutine = false;  // reverts to True if at least one component still running
      for (const thisComponent of lossDirsComponents)
        if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
          continueRoutine = true;
          break;
        }
      
      // refresh the screen if continuing
      if (continueRoutine) {
        return Scheduler.Event.FLIP_REPEAT;
      } else {
        return Scheduler.Event.NEXT;
      }
    };
  }
  
  function lossDirsRoutineEnd(snapshot) {
    return async function () {
      //--- Ending Routine 'lossDirs' ---
      for (const thisComponent of lossDirsComponents) {
        if (typeof thisComponent.setAutoDraw === 'function') {
          thisComponent.setAutoDraw(false);
        }
      }
      psychoJS.experiment.addData('lossDirs.stopped', globalClock.getTime());
      // update the trial handler
      if (currentLoop instanceof MultiStairHandler) {
        currentLoop.addResponse(advanceScreen4.corr, level);
      }
      psychoJS.experiment.addData('advanceScreen4.keys', advanceScreen4.keys);
      if (typeof advanceScreen4.keys !== 'undefined') {  // we had a response
          psychoJS.experiment.addData('advanceScreen4.rt', advanceScreen4.rt);
          psychoJS.experiment.addData('advanceScreen4.duration', advanceScreen4.duration);
          routineTimer.reset();
          }
      
      advanceScreen4.stop();
      // Run 'End Routine' code from framePracOn
      frame.setAutoDraw(true);
      
      // the Routine "lossDirs" was not non-slip safe, so reset the non-slip timer
      routineTimer.reset();
      
      // Routines running outside a loop should always advance the datafile row
      if (currentLoop === psychoJS.experiment) {
        psychoJS.experiment.nextEntry(snapshot);
      }
      return Scheduler.Event.NEXT;
    }
  }
  
  function prac2CueRoutineBegin(snapshot) {
    return async function () {
      TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
      
      //--- Prepare to start Routine 'prac2Cue' ---
      t = 0;
      frameN = -1;
      continueRoutine = true; // until we're told otherwise
      // keep track of whether this Routine was forcibly ended
      routineForceEnded = false;
      prac2CueClock.reset(routineTimer.getTime());
      routineTimer.add(3.000000);
      prac2CueMaxDurationReached = false;
      // update component parameters for each repeat
      prac2Resp.keys = undefined;
      prac2Resp.rt = undefined;
      _prac2Resp_allKeys = [];
      // Run 'Begin Routine' code from setCuesPrac
      /* Syntax Error: Fix Python code */
      // Run 'Begin Routine' code from pracSelect
      cueEndTime = 3;
      selectionIndicator.status = NOT_STARTED;
      
      // Run 'Begin Routine' code from pracFrameLabels
      if ((condition === "lowgain")) {
          corrvalue = 0.2;
          incorrvalue = 0;
          topLabel = `+$${Math.abs(corrvalue)}`;
          bottomLabel = `+$${Math.abs(incorrvalue)}`;
      }
      if ((condition === "highgain")) {
          corrvalue = 1.0;
          incorrvalue = 0;
          topLabel = `+$${Math.abs(corrvalue)}`;
          bottomLabel = `+$${Math.abs(incorrvalue)}`;
      }
      if ((condition === "lowloss")) {
          corrvalue = 0;
          incorrvalue = (- 0.2);
          topLabel = `-$${Math.abs(corrvalue)}`;
          bottomLabel = `-$${Math.abs(incorrvalue)}`;
      }
      if ((condition === "highloss")) {
          corrvalue = 0;
          incorrvalue = (- 1.0);
          topLabel = `-$${Math.abs(corrvalue)}`;
          bottomLabel = `-$${Math.abs(incorrvalue)}`;
      }
      topFrameText.setText(topLabel);
      topFrameText.status = NOT_STARTED;
      bottomFrameText.setText(bottomLabel);
      bottomFrameText.status = NOT_STARTED;
      
      psychoJS.experiment.addData('prac2Cue.started', globalClock.getTime());
      prac2CueMaxDuration = null
      // keep track of which components have finished
      prac2CueComponents = [];
      prac2CueComponents.push(prac2LeftCue);
      prac2CueComponents.push(prac2RightCue);
      prac2CueComponents.push(prac2Resp);
      prac2CueComponents.push(ISI);
      
      for (const thisComponent of prac2CueComponents)
        if ('status' in thisComponent)
          thisComponent.status = PsychoJS.Status.NOT_STARTED;
      return Scheduler.Event.NEXT;
    }
  }
  
  function prac2CueRoutineEachFrame() {
    return async function () {
      //--- Loop for each frame of Routine 'prac2Cue' ---
      // get current time
      t = prac2CueClock.getTime();
      frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
      // update/draw components on each frame
      
      // *prac2LeftCue* updates
      if (t >= 0.5 && prac2LeftCue.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        prac2LeftCue.tStart = t;  // (not accounting for frame time here)
        prac2LeftCue.frameNStart = frameN;  // exact frame index
        
        prac2LeftCue.setAutoDraw(true);
      }
      
      
      // if prac2LeftCue is active this frame...
      if (prac2LeftCue.status === PsychoJS.Status.STARTED) {
      }
      
      frameRemains = 0.5 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
      if (prac2LeftCue.status === PsychoJS.Status.STARTED && t >= frameRemains) {
        // keep track of stop time/frame for later
        prac2LeftCue.tStop = t;  // not accounting for scr refresh
        prac2LeftCue.frameNStop = frameN;  // exact frame index
        // update status
        prac2LeftCue.status = PsychoJS.Status.FINISHED;
        prac2LeftCue.setAutoDraw(false);
      }
      
      
      // *prac2RightCue* updates
      if (t >= 0.5 && prac2RightCue.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        prac2RightCue.tStart = t;  // (not accounting for frame time here)
        prac2RightCue.frameNStart = frameN;  // exact frame index
        
        prac2RightCue.setAutoDraw(true);
      }
      
      
      // if prac2RightCue is active this frame...
      if (prac2RightCue.status === PsychoJS.Status.STARTED) {
      }
      
      frameRemains = 0.5 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
      if (prac2RightCue.status === PsychoJS.Status.STARTED && t >= frameRemains) {
        // keep track of stop time/frame for later
        prac2RightCue.tStop = t;  // not accounting for scr refresh
        prac2RightCue.frameNStop = frameN;  // exact frame index
        // update status
        prac2RightCue.status = PsychoJS.Status.FINISHED;
        prac2RightCue.setAutoDraw(false);
      }
      
      
      // *prac2Resp* updates
      if (t >= 0.5 && prac2Resp.status === PsychoJS.Status.NOT_STARTED) {
        // keep track of start time/frame for later
        prac2Resp.tStart = t;  // (not accounting for frame time here)
        prac2Resp.frameNStart = frameN;  // exact frame index
        
        // keyboard checking is just starting
        psychoJS.window.callOnFlip(function() { prac2Resp.clock.reset(); });  // t=0 on next screen flip
        psychoJS.window.callOnFlip(function() { prac2Resp.start(); }); // start on screen flip
        psychoJS.window.callOnFlip(function() { prac2Resp.clearEvents(); });
      }
      frameRemains = 0.5 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
      if (prac2Resp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
        // keep track of stop time/frame for later
        prac2Resp.tStop = t;  // not accounting for scr refresh
        prac2Resp.frameNStop = frameN;  // exact frame index
        // update status
        prac2Resp.status = PsychoJS.Status.FINISHED;
        frameRemains = 0.5 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
        if (prac2Resp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
          // keep track of stop time/frame for later
          prac2Resp.tStop = t;  // not accounting for scr refresh
          prac2Resp.frameNStop = frameN;  // exact frame index
          // update status
          prac2Resp.status = PsychoJS.Status.FINISHED;
          prac2Resp.status = PsychoJS.Status.FINISHED;
            }
          
        }
        
        // if prac2Resp is active this frame...
        if (prac2Resp.status === PsychoJS.Status.STARTED) {
          let theseKeys = prac2Resp.getKeys({keyList: allowedKeys, waitRelease: false});
          _prac2Resp_allKeys = _prac2Resp_allKeys.concat(theseKeys);
          if (_prac2Resp_allKeys.length > 0) {
            prac2Resp.keys = _prac2Resp_allKeys[0].name;  // just the first key pressed
            prac2Resp.rt = _prac2Resp_allKeys[0].rt;
            prac2Resp.duration = _prac2Resp_allKeys[0].duration;
            // was this correct?
            if (prac2Resp.keys == correct) {
                prac2Resp.corr = 1;
            } else {
                prac2Resp.corr = 0;
            }
          }
        }
        
        // Run 'Each Frame' code from pracSelect
        if ((t > cueEndTime)) {
            continueRoutine = false;
        }
        if ((prac2Resp.keys && (selectionIndicator.status === NOT_STARTED))) {
            selectionIndicator.status = PsychoJS.Status.STARTED;
            selectionIndicator.setPos(selectionPosition(prac2Resp.keys));
            selectionIndicator.setAutoDraw(true);
        }
        
        // Run 'Each Frame' code from pracFrameLabels
        if (((t > 0.5) && (topFrameText.status === NOT_STARTED))) {
            topFrameText.setAutoDraw(true);
            bottomFrameText.setAutoDraw(true);
            topFrameText.status = PsychoJS.Status.STARTED;
        }
        
        if (t >= 0.0 && ISI.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          ISI.tStart = t;  // (not accounting for frame time here)
          ISI.frameNStart = frameN;  // exact frame index
          
          ISI.status = PsychoJS.Status.STARTED;
          // Updating other components during *ISI*
          console.log('register and start downloading resources specified by component prac2LeftCue');
          await psychoJS.serverManager.prepareResources(leftImage);
          ISI.status = PsychoJS.Status.STARTED;
          prac2LeftCue.setImage(leftImage);
          console.log('register and start downloading resources specified by component prac2RightCue');
          await psychoJS.serverManager.prepareResources(rightImage);
          ISI.status = PsychoJS.Status.STARTED;
          prac2RightCue.setImage(rightImage);
          // Component updates done
        }
        frameRemains = 0.0 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
        if (ISI.status === PsychoJS.Status.STARTED && t >= frameRemains) {
          // keep track of stop time/frame for later
          ISI.tStop = t;  // not accounting for scr refresh
          ISI.frameNStop = frameN;  // exact frame index
          // update status
          ISI.status = PsychoJS.Status.FINISHED;
          if (psychoJS.serverManager.getResourceStatus(leftImage) === core.ServerManager.ResourceStatus.DOWNLOADED) {
            console.log('finished downloading resources specified by component ISI');
          } else {
            console.log('resource specified in ISI took longer than expected to download');
            await waitForResources(resources = leftImage)
          }
          if (psychoJS.serverManager.getResourceStatus(rightImage) === core.ServerManager.ResourceStatus.DOWNLOADED) {
            console.log('finished downloading resources specified by component ISI');
          } else {
            console.log('resource specified in ISI took longer than expected to download');
            await waitForResources(resources = rightImage)
          }
          ISI.status = PsychoJS.Status.FINISHED;
        }
        // check for quit (typically the Esc key)
        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
          return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }
        
        // check if the Routine should terminate
        if (!continueRoutine) {  // a component has requested a forced-end of Routine
          routineForceEnded = true;
          return Scheduler.Event.NEXT;
        }
        
        continueRoutine = false;  // reverts to True if at least one component still running
        for (const thisComponent of prac2CueComponents)
          if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
            continueRoutine = true;
            break;
          }
        
        // refresh the screen if continuing
        if (continueRoutine && routineTimer.getTime() > 0) {
          return Scheduler.Event.FLIP_REPEAT;
        } else {
          return Scheduler.Event.NEXT;
        }
      };
    }
    
    function prac2CueRoutineEnd(snapshot) {
      return async function () {
        //--- Ending Routine 'prac2Cue' ---
        for (const thisComponent of prac2CueComponents) {
          if (typeof thisComponent.setAutoDraw === 'function') {
            thisComponent.setAutoDraw(false);
          }
        }
        psychoJS.experiment.addData('prac2Cue.stopped', globalClock.getTime());
        // was no response the correct answer?!
        if (prac2Resp.keys === undefined) {
          if (['None','none',undefined].includes(correct)) {
             prac2Resp.corr = 1;  // correct non-response
          } else {
             prac2Resp.corr = 0;  // failed to respond (incorrectly)
          }
        }
        // store data for current loop
        // update the trial handler
        if (currentLoop instanceof MultiStairHandler) {
          currentLoop.addResponse(prac2Resp.corr, level);
        }
        psychoJS.experiment.addData('prac2Resp.keys', prac2Resp.keys);
        psychoJS.experiment.addData('prac2Resp.corr', prac2Resp.corr);
        if (typeof prac2Resp.keys !== 'undefined') {  // we had a response
            psychoJS.experiment.addData('prac2Resp.rt', prac2Resp.rt);
            psychoJS.experiment.addData('prac2Resp.duration', prac2Resp.duration);
            }
        
        prac2Resp.stop();
        // Run 'End Routine' code from setCuesPrac
        if ((prac2Resp.keys === responseKeys["left"])) {
            currentLoop.addData("optedSide", "left");
            currentLoop.addData("optedImg", leftImage);
        } else {
            if ((prac2Resp.keys === responseKeys["right"])) {
                currentLoop.addData("optedSide", "right");
                currentLoop.addData("optedImg", rightImage);
            } else {
                currentLoop.addData("optedSide", "NA");
                currentLoop.addData("optedImg", "NA");
            }
        }
        
        // Run 'End Routine' code from pracSelect
        selectionIndicator.setAutoDraw(false);
        
        // Run 'End Routine' code from pracFrameLabels
        topFrameText.status = NOT_STARTED;
        
        if (routineForceEnded) {
            routineTimer.reset();} else if (prac2CueMaxDurationReached) {
            prac2CueClock.add(prac2CueMaxDuration);
        } else {
            prac2CueClock.add(3.000000);
        }
        // Routines running outside a loop should always advance the datafile row
        if (currentLoop === psychoJS.experiment) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        return Scheduler.Event.NEXT;
      }
    }
    
    function prac2JitterRoutineBegin(snapshot) {
      return async function () {
        TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
        
        //--- Prepare to start Routine 'prac2Jitter' ---
        t = 0;
        frameN = -1;
        continueRoutine = true; // until we're told otherwise
        // keep track of whether this Routine was forcibly ended
        routineForceEnded = false;
        prac2JitterClock.reset();
        routineTimer.reset();
        prac2JitterMaxDurationReached = false;
        // update component parameters for each repeat
        psychoJS.experiment.addData('prac2Jitter.started', globalClock.getTime());
        prac2JitterMaxDuration = null
        // keep track of which components have finished
        prac2JitterComponents = [];
        prac2JitterComponents.push(prac2Fix);
        
        for (const thisComponent of prac2JitterComponents)
          if ('status' in thisComponent)
            thisComponent.status = PsychoJS.Status.NOT_STARTED;
        return Scheduler.Event.NEXT;
      }
    }
    
    function prac2JitterRoutineEachFrame() {
      return async function () {
        //--- Loop for each frame of Routine 'prac2Jitter' ---
        // get current time
        t = prac2JitterClock.getTime();
        frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
        // update/draw components on each frame
        
        // *prac2Fix* updates
        if (t >= 0.0 && prac2Fix.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          prac2Fix.tStart = t;  // (not accounting for frame time here)
          prac2Fix.frameNStart = frameN;  // exact frame index
          
          prac2Fix.setAutoDraw(true);
        }
        
        
        // if prac2Fix is active this frame...
        if (prac2Fix.status === PsychoJS.Status.STARTED) {
        }
        
        frameRemains = 0.0 + outcomeDelay - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
        if (prac2Fix.status === PsychoJS.Status.STARTED && t >= frameRemains) {
          // keep track of stop time/frame for later
          prac2Fix.tStop = t;  // not accounting for scr refresh
          prac2Fix.frameNStop = frameN;  // exact frame index
          // update status
          prac2Fix.status = PsychoJS.Status.FINISHED;
          prac2Fix.setAutoDraw(false);
        }
        
        // check for quit (typically the Esc key)
        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
          return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }
        
        // check if the Routine should terminate
        if (!continueRoutine) {  // a component has requested a forced-end of Routine
          routineForceEnded = true;
          return Scheduler.Event.NEXT;
        }
        
        continueRoutine = false;  // reverts to True if at least one component still running
        for (const thisComponent of prac2JitterComponents)
          if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
            continueRoutine = true;
            break;
          }
        
        // refresh the screen if continuing
        if (continueRoutine) {
          return Scheduler.Event.FLIP_REPEAT;
        } else {
          return Scheduler.Event.NEXT;
        }
      };
    }
    
    function prac2JitterRoutineEnd(snapshot) {
      return async function () {
        //--- Ending Routine 'prac2Jitter' ---
        for (const thisComponent of prac2JitterComponents) {
          if (typeof thisComponent.setAutoDraw === 'function') {
            thisComponent.setAutoDraw(false);
          }
        }
        psychoJS.experiment.addData('prac2Jitter.stopped', globalClock.getTime());
        // the Routine "prac2Jitter" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset();
        
        // Routines running outside a loop should always advance the datafile row
        if (currentLoop === psychoJS.experiment) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        return Scheduler.Event.NEXT;
      }
    }
    
    function prac2OutRoutineBegin(snapshot) {
      return async function () {
        TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
        
        //--- Prepare to start Routine 'prac2Out' ---
        t = 0;
        frameN = -1;
        continueRoutine = true; // until we're told otherwise
        // keep track of whether this Routine was forcibly ended
        routineForceEnded = false;
        prac2OutClock.reset(routineTimer.getTime());
        routineTimer.add(1.000000);
        prac2OutMaxDurationReached = false;
        // update component parameters for each repeat
        // Run 'Begin Routine' code from pracRightChoice
        if ((prac2Resp.keys === sidePositions[optimalImg])) {
            Choice = "optimal";
        } else {
            if ((prac2Resp.keys === sidePositions[suboptimalImg])) {
                Choice = "suboptimal";
            } else {
                Choice = "NA";
            }
        }
        Accuracy = prac2Resp.corr;
        currentLoop.addData("optedFor", Choice);
        currentLoop.addData("accuracy", Accuracy);
        
        // Run 'Begin Routine' code from definePracOut
        if ((condition === "highgain")) {
            trueOperator = "+";
            falseOperator = "+";
        } else {
            if ((condition === "lowgain")) {
                trueOperator = "+";
                falseOperator = "+";
            } else {
                if ((condition === "highloss")) {
                    trueOperator = "-";
                    falseOperator = "-";
                } else {
                    trueOperator = "-";
                    falseOperator = "-";
                }
            }
        }
        if (prac2Resp.keys) {
            outcome = choose_prac_outcome_from_cue(prac2Resp);
            currentLoop.addData("outcome", outcome);
            if (outcome) {
                change = corrvalue;
                outcome_msg = `${trueOperator}$${Math.abs(change)}`;
                outcome_image = null;
                Feedback = 1;
            } else {
                change = incorrvalue;
                outcome_msg = `${falseOperator}$${Math.abs(change)}`;
                outcome_image = null;
                Feedback = 0;
            }
        } else {
            change = incorrvalue;
            currentLoop.addData("outcome", null);
            outcome_msg = `${falseOperator}$${Math.abs(change)}`;
            outcome_image = null;
            Feedback = "NA";
        }
        bank += change;
        currentLoop.addData("Feedback", Feedback);
        currentLoop.addData("gainLossAmount", change);
        currentLoop.addData("runningBankTotal", bank);
        currentLoop.addData("feedbackTime", fmriClock.getTime());
        
        outcomePracText.setText(outcome_msg);
        psychoJS.experiment.addData('prac2Out.started', globalClock.getTime());
        prac2OutMaxDuration = null
        // keep track of which components have finished
        prac2OutComponents = [];
        prac2OutComponents.push(outcomePracText);
        
        for (const thisComponent of prac2OutComponents)
          if ('status' in thisComponent)
            thisComponent.status = PsychoJS.Status.NOT_STARTED;
        return Scheduler.Event.NEXT;
      }
    }
    
    function prac2OutRoutineEachFrame() {
      return async function () {
        //--- Loop for each frame of Routine 'prac2Out' ---
        // get current time
        t = prac2OutClock.getTime();
        frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
        // update/draw components on each frame
        
        // *outcomePracText* updates
        if (t >= 0.0 && outcomePracText.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          outcomePracText.tStart = t;  // (not accounting for frame time here)
          outcomePracText.frameNStart = frameN;  // exact frame index
          
          outcomePracText.setAutoDraw(true);
        }
        
        
        // if outcomePracText is active this frame...
        if (outcomePracText.status === PsychoJS.Status.STARTED) {
        }
        
        frameRemains = 0.0 + 1.0 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
        if (outcomePracText.status === PsychoJS.Status.STARTED && t >= frameRemains) {
          // keep track of stop time/frame for later
          outcomePracText.tStop = t;  // not accounting for scr refresh
          outcomePracText.frameNStop = frameN;  // exact frame index
          // update status
          outcomePracText.status = PsychoJS.Status.FINISHED;
          outcomePracText.setAutoDraw(false);
        }
        
        // check for quit (typically the Esc key)
        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
          return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }
        
        // check if the Routine should terminate
        if (!continueRoutine) {  // a component has requested a forced-end of Routine
          routineForceEnded = true;
          return Scheduler.Event.NEXT;
        }
        
        continueRoutine = false;  // reverts to True if at least one component still running
        for (const thisComponent of prac2OutComponents)
          if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
            continueRoutine = true;
            break;
          }
        
        // refresh the screen if continuing
        if (continueRoutine && routineTimer.getTime() > 0) {
          return Scheduler.Event.FLIP_REPEAT;
        } else {
          return Scheduler.Event.NEXT;
        }
      };
    }
    
    function prac2OutRoutineEnd(snapshot) {
      return async function () {
        //--- Ending Routine 'prac2Out' ---
        for (const thisComponent of prac2OutComponents) {
          if (typeof thisComponent.setAutoDraw === 'function') {
            thisComponent.setAutoDraw(false);
          }
        }
        psychoJS.experiment.addData('prac2Out.stopped', globalClock.getTime());
        // Run 'End Routine' code from definePracOut
        expInfo["bank"] = bank;
        currentLoop.addData("feedbackOffTime", fmriClock.getTime());
        
        // Run 'End Routine' code from frameOff
        topFrameText.setAutoDraw(false);
        bottomFrameText.setAutoDraw(false);
        
        if (routineForceEnded) {
            routineTimer.reset();} else if (prac2OutMaxDurationReached) {
            prac2OutClock.add(prac2OutMaxDuration);
        } else {
            prac2OutClock.add(1.000000);
        }
        // Routines running outside a loop should always advance the datafile row
        if (currentLoop === psychoJS.experiment) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        return Scheduler.Event.NEXT;
      }
    }
    
    function check3RoutineBegin(snapshot) {
      return async function () {
        TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
        
        //--- Prepare to start Routine 'check3' ---
        t = 0;
        frameN = -1;
        continueRoutine = true; // until we're told otherwise
        // keep track of whether this Routine was forcibly ended
        routineForceEnded = false;
        check3Clock.reset();
        routineTimer.reset();
        check3MaxDurationReached = false;
        // update component parameters for each repeat
        // Run 'Begin Routine' code from check3Select
        frame.setAutoDraw(false);
        selectionIndicatorPrac.status = NOT_STARTED;
        t_started = false;
        
        check3Press.keys = undefined;
        check3Press.rt = undefined;
        _check3Press_allKeys = [];
        down2.keys = undefined;
        down2.rt = undefined;
        _down2_allKeys = [];
        psychoJS.experiment.addData('check3.started', globalClock.getTime());
        check3MaxDuration = null
        // keep track of which components have finished
        check3Components = [];
        check3Components.push(check3Text);
        check3Components.push(check3ImageLeft);
        check3Components.push(check3ImageRight);
        check3Components.push(check3Press);
        check3Components.push(down2);
        
        for (const thisComponent of check3Components)
          if ('status' in thisComponent)
            thisComponent.status = PsychoJS.Status.NOT_STARTED;
        return Scheduler.Event.NEXT;
      }
    }
    
    function check3RoutineEachFrame() {
      return async function () {
        //--- Loop for each frame of Routine 'check3' ---
        // get current time
        t = check3Clock.getTime();
        frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
        // update/draw components on each frame
        
        // *check3Text* updates
        if (t >= 0.0 && check3Text.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check3Text.tStart = t;  // (not accounting for frame time here)
          check3Text.frameNStart = frameN;  // exact frame index
          
          check3Text.setAutoDraw(true);
        }
        
        
        // if check3Text is active this frame...
        if (check3Text.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *check3ImageLeft* updates
        if (t >= 0.0 && check3ImageLeft.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check3ImageLeft.tStart = t;  // (not accounting for frame time here)
          check3ImageLeft.frameNStart = frameN;  // exact frame index
          
          check3ImageLeft.setAutoDraw(true);
        }
        
        
        // if check3ImageLeft is active this frame...
        if (check3ImageLeft.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *check3ImageRight* updates
        if (t >= 0.0 && check3ImageRight.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check3ImageRight.tStart = t;  // (not accounting for frame time here)
          check3ImageRight.frameNStart = frameN;  // exact frame index
          
          check3ImageRight.setAutoDraw(true);
        }
        
        
        // if check3ImageRight is active this frame...
        if (check3ImageRight.status === PsychoJS.Status.STARTED) {
        }
        
        // Run 'Each Frame' code from check3Select
        if (t_started) {
            if ((t > (t_started + 5))) {
                continueRoutine = false;
            }
        }
        if ((check3Press.keys && (selectionIndicatorPrac.status === NOT_STARTED))) {
            selectionIndicatorPrac.status = PsychoJS.Status.STARTED;
            selectionIndicatorPrac.setPos(selectionPosition(check3Press.keys));
            selectionIndicatorPrac.setAutoDraw(true);
            t_started = new util.Clock().getTime();
        }
        
        
        // *check3Press* updates
        if (t >= 0.0 && check3Press.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check3Press.tStart = t;  // (not accounting for frame time here)
          check3Press.frameNStart = frameN;  // exact frame index
          
          // keyboard checking is just starting
          psychoJS.window.callOnFlip(function() { check3Press.clock.reset(); });  // t=0 on next screen flip
          psychoJS.window.callOnFlip(function() { check3Press.start(); }); // start on screen flip
          psychoJS.window.callOnFlip(function() { check3Press.clearEvents(); });
        }
        
        // if check3Press is active this frame...
        if (check3Press.status === PsychoJS.Status.STARTED) {
          let theseKeys = check3Press.getKeys({keyList: allowedKeys, waitRelease: false});
          _check3Press_allKeys = _check3Press_allKeys.concat(theseKeys);
          if (_check3Press_allKeys.length > 0) {
            check3Press.keys = _check3Press_allKeys[_check3Press_allKeys.length - 1].name;  // just the last key pressed
            check3Press.rt = _check3Press_allKeys[_check3Press_allKeys.length - 1].rt;
            check3Press.duration = _check3Press_allKeys[_check3Press_allKeys.length - 1].duration;
          }
        }
        
        
        // *down2* updates
        if (t >= 0.0 && down2.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          down2.tStart = t;  // (not accounting for frame time here)
          down2.frameNStart = frameN;  // exact frame index
          
          // keyboard checking is just starting
          psychoJS.window.callOnFlip(function() { down2.clock.reset(); });  // t=0 on next screen flip
          psychoJS.window.callOnFlip(function() { down2.start(); }); // start on screen flip
          psychoJS.window.callOnFlip(function() { down2.clearEvents(); });
        }
        
        // if down2 is active this frame...
        if (down2.status === PsychoJS.Status.STARTED) {
          let theseKeys = down2.getKeys({keyList: 'down', waitRelease: false});
          _down2_allKeys = _down2_allKeys.concat(theseKeys);
          if (_down2_allKeys.length > 0) {
            down2.keys = _down2_allKeys[_down2_allKeys.length - 1].name;  // just the last key pressed
            down2.rt = _down2_allKeys[_down2_allKeys.length - 1].rt;
            down2.duration = _down2_allKeys[_down2_allKeys.length - 1].duration;
            // a response ends the routine
            continueRoutine = false;
          }
        }
        
        // check for quit (typically the Esc key)
        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
          return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }
        
        // check if the Routine should terminate
        if (!continueRoutine) {  // a component has requested a forced-end of Routine
          routineForceEnded = true;
          return Scheduler.Event.NEXT;
        }
        
        continueRoutine = false;  // reverts to True if at least one component still running
        for (const thisComponent of check3Components)
          if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
            continueRoutine = true;
            break;
          }
        
        // refresh the screen if continuing
        if (continueRoutine) {
          return Scheduler.Event.FLIP_REPEAT;
        } else {
          return Scheduler.Event.NEXT;
        }
      };
    }
    
    function check3RoutineEnd(snapshot) {
      return async function () {
        //--- Ending Routine 'check3' ---
        for (const thisComponent of check3Components) {
          if (typeof thisComponent.setAutoDraw === 'function') {
            thisComponent.setAutoDraw(false);
          }
        }
        psychoJS.experiment.addData('check3.stopped', globalClock.getTime());
        // Run 'End Routine' code from check3Select
        selectionIndicatorPrac.setAutoDraw(false);
        if ((check3Press.keys === "left")) {
            currentLoop.addData("check3", "correct");
        } else {
            currentLoop.addData("check3", "incorrect");
        }
        
        // update the trial handler
        if (currentLoop instanceof MultiStairHandler) {
          currentLoop.addResponse(check3Press.corr, level);
        }
        psychoJS.experiment.addData('check3Press.keys', check3Press.keys);
        if (typeof check3Press.keys !== 'undefined') {  // we had a response
            psychoJS.experiment.addData('check3Press.rt', check3Press.rt);
            psychoJS.experiment.addData('check3Press.duration', check3Press.duration);
            }
        
        check3Press.stop();
        // update the trial handler
        if (currentLoop instanceof MultiStairHandler) {
          currentLoop.addResponse(down2.corr, level);
        }
        psychoJS.experiment.addData('down2.keys', down2.keys);
        if (typeof down2.keys !== 'undefined') {  // we had a response
            psychoJS.experiment.addData('down2.rt', down2.rt);
            psychoJS.experiment.addData('down2.duration', down2.duration);
            routineTimer.reset();
            }
        
        down2.stop();
        // the Routine "check3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset();
        
        // Routines running outside a loop should always advance the datafile row
        if (currentLoop === psychoJS.experiment) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        return Scheduler.Event.NEXT;
      }
    }
    
    function check2RoutineBegin(snapshot) {
      return async function () {
        TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
        
        //--- Prepare to start Routine 'check2' ---
        t = 0;
        frameN = -1;
        continueRoutine = true; // until we're told otherwise
        // keep track of whether this Routine was forcibly ended
        routineForceEnded = false;
        check2Clock.reset();
        routineTimer.reset();
        check2MaxDurationReached = false;
        // update component parameters for each repeat
        // Run 'Begin Routine' code from check2Select
        selectionIndicatorPrac.status = NOT_STARTED;
        t_started = false;
        
        check2Press.keys = undefined;
        check2Press.rt = undefined;
        _check2Press_allKeys = [];
        down3.keys = undefined;
        down3.rt = undefined;
        _down3_allKeys = [];
        psychoJS.experiment.addData('check2.started', globalClock.getTime());
        check2MaxDuration = null
        // keep track of which components have finished
        check2Components = [];
        check2Components.push(check2Text);
        check2Components.push(check2ImageLeft);
        check2Components.push(check2ImageRight);
        check2Components.push(check2Press);
        check2Components.push(down3);
        
        for (const thisComponent of check2Components)
          if ('status' in thisComponent)
            thisComponent.status = PsychoJS.Status.NOT_STARTED;
        return Scheduler.Event.NEXT;
      }
    }
    
    function check2RoutineEachFrame() {
      return async function () {
        //--- Loop for each frame of Routine 'check2' ---
        // get current time
        t = check2Clock.getTime();
        frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
        // update/draw components on each frame
        
        // *check2Text* updates
        if (t >= 0.0 && check2Text.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check2Text.tStart = t;  // (not accounting for frame time here)
          check2Text.frameNStart = frameN;  // exact frame index
          
          check2Text.setAutoDraw(true);
        }
        
        
        // if check2Text is active this frame...
        if (check2Text.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *check2ImageLeft* updates
        if (t >= 0.0 && check2ImageLeft.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check2ImageLeft.tStart = t;  // (not accounting for frame time here)
          check2ImageLeft.frameNStart = frameN;  // exact frame index
          
          check2ImageLeft.setAutoDraw(true);
        }
        
        
        // if check2ImageLeft is active this frame...
        if (check2ImageLeft.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *check2ImageRight* updates
        if (t >= 0.0 && check2ImageRight.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check2ImageRight.tStart = t;  // (not accounting for frame time here)
          check2ImageRight.frameNStart = frameN;  // exact frame index
          
          check2ImageRight.setAutoDraw(true);
        }
        
        
        // if check2ImageRight is active this frame...
        if (check2ImageRight.status === PsychoJS.Status.STARTED) {
        }
        
        // Run 'Each Frame' code from check2Select
        if (t_started) {
            if ((t > (t_started + 5))) {
                continueRoutine = false;
            }
        }
        if ((check2Press.keys && (selectionIndicatorPrac.status === NOT_STARTED))) {
            selectionIndicatorPrac.status = PsychoJS.Status.STARTED;
            selectionIndicatorPrac.setPos(selectionPosition(check2Press.keys));
            selectionIndicatorPrac.setAutoDraw(true);
            t_started = new util.Clock().getTime();
        }
        
        
        // *check2Press* updates
        if (t >= 0.0 && check2Press.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check2Press.tStart = t;  // (not accounting for frame time here)
          check2Press.frameNStart = frameN;  // exact frame index
          
          // keyboard checking is just starting
          psychoJS.window.callOnFlip(function() { check2Press.clock.reset(); });  // t=0 on next screen flip
          psychoJS.window.callOnFlip(function() { check2Press.start(); }); // start on screen flip
          psychoJS.window.callOnFlip(function() { check2Press.clearEvents(); });
        }
        
        // if check2Press is active this frame...
        if (check2Press.status === PsychoJS.Status.STARTED) {
          let theseKeys = check2Press.getKeys({keyList: allowedKeys, waitRelease: false});
          _check2Press_allKeys = _check2Press_allKeys.concat(theseKeys);
          if (_check2Press_allKeys.length > 0) {
            check2Press.keys = _check2Press_allKeys[_check2Press_allKeys.length - 1].name;  // just the last key pressed
            check2Press.rt = _check2Press_allKeys[_check2Press_allKeys.length - 1].rt;
            check2Press.duration = _check2Press_allKeys[_check2Press_allKeys.length - 1].duration;
          }
        }
        
        
        // *down3* updates
        if (t >= 0.0 && down3.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          down3.tStart = t;  // (not accounting for frame time here)
          down3.frameNStart = frameN;  // exact frame index
          
          // keyboard checking is just starting
          psychoJS.window.callOnFlip(function() { down3.clock.reset(); });  // t=0 on next screen flip
          psychoJS.window.callOnFlip(function() { down3.start(); }); // start on screen flip
          psychoJS.window.callOnFlip(function() { down3.clearEvents(); });
        }
        
        // if down3 is active this frame...
        if (down3.status === PsychoJS.Status.STARTED) {
          let theseKeys = down3.getKeys({keyList: 'down', waitRelease: false});
          _down3_allKeys = _down3_allKeys.concat(theseKeys);
          if (_down3_allKeys.length > 0) {
            down3.keys = _down3_allKeys[_down3_allKeys.length - 1].name;  // just the last key pressed
            down3.rt = _down3_allKeys[_down3_allKeys.length - 1].rt;
            down3.duration = _down3_allKeys[_down3_allKeys.length - 1].duration;
            // a response ends the routine
            continueRoutine = false;
          }
        }
        
        // check for quit (typically the Esc key)
        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
          return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }
        
        // check if the Routine should terminate
        if (!continueRoutine) {  // a component has requested a forced-end of Routine
          routineForceEnded = true;
          return Scheduler.Event.NEXT;
        }
        
        continueRoutine = false;  // reverts to True if at least one component still running
        for (const thisComponent of check2Components)
          if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
            continueRoutine = true;
            break;
          }
        
        // refresh the screen if continuing
        if (continueRoutine) {
          return Scheduler.Event.FLIP_REPEAT;
        } else {
          return Scheduler.Event.NEXT;
        }
      };
    }
    
    function check2RoutineEnd(snapshot) {
      return async function () {
        //--- Ending Routine 'check2' ---
        for (const thisComponent of check2Components) {
          if (typeof thisComponent.setAutoDraw === 'function') {
            thisComponent.setAutoDraw(false);
          }
        }
        psychoJS.experiment.addData('check2.stopped', globalClock.getTime());
        // Run 'End Routine' code from check2Select
        selectionIndicatorPrac.setAutoDraw(false);
        if ((check2Press.keys === "right")) {
            currentLoop.addData("check2", "correct");
        } else {
            currentLoop.addData("check2", "incorrect");
        }
        
        // update the trial handler
        if (currentLoop instanceof MultiStairHandler) {
          currentLoop.addResponse(check2Press.corr, level);
        }
        psychoJS.experiment.addData('check2Press.keys', check2Press.keys);
        if (typeof check2Press.keys !== 'undefined') {  // we had a response
            psychoJS.experiment.addData('check2Press.rt', check2Press.rt);
            psychoJS.experiment.addData('check2Press.duration', check2Press.duration);
            }
        
        check2Press.stop();
        // update the trial handler
        if (currentLoop instanceof MultiStairHandler) {
          currentLoop.addResponse(down3.corr, level);
        }
        psychoJS.experiment.addData('down3.keys', down3.keys);
        if (typeof down3.keys !== 'undefined') {  // we had a response
            psychoJS.experiment.addData('down3.rt', down3.rt);
            psychoJS.experiment.addData('down3.duration', down3.duration);
            routineTimer.reset();
            }
        
        down3.stop();
        // the Routine "check2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset();
        
        // Routines running outside a loop should always advance the datafile row
        if (currentLoop === psychoJS.experiment) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        return Scheduler.Event.NEXT;
      }
    }
    
    function check4RoutineBegin(snapshot) {
      return async function () {
        TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
        
        //--- Prepare to start Routine 'check4' ---
        t = 0;
        frameN = -1;
        continueRoutine = true; // until we're told otherwise
        // keep track of whether this Routine was forcibly ended
        routineForceEnded = false;
        check4Clock.reset();
        routineTimer.reset();
        check4MaxDurationReached = false;
        // update component parameters for each repeat
        // Run 'Begin Routine' code from check4Select
        selectionIndicatorPrac.status = NOT_STARTED;
        t_started = false;
        
        check4Press.keys = undefined;
        check4Press.rt = undefined;
        _check4Press_allKeys = [];
        down4.keys = undefined;
        down4.rt = undefined;
        _down4_allKeys = [];
        psychoJS.experiment.addData('check4.started', globalClock.getTime());
        check4MaxDuration = null
        // keep track of which components have finished
        check4Components = [];
        check4Components.push(check4Text);
        check4Components.push(check4ImageLeft);
        check4Components.push(check4ImageRight);
        check4Components.push(check4Press);
        check4Components.push(down4);
        
        for (const thisComponent of check4Components)
          if ('status' in thisComponent)
            thisComponent.status = PsychoJS.Status.NOT_STARTED;
        return Scheduler.Event.NEXT;
      }
    }
    
    function check4RoutineEachFrame() {
      return async function () {
        //--- Loop for each frame of Routine 'check4' ---
        // get current time
        t = check4Clock.getTime();
        frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
        // update/draw components on each frame
        
        // *check4Text* updates
        if (t >= 0.0 && check4Text.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check4Text.tStart = t;  // (not accounting for frame time here)
          check4Text.frameNStart = frameN;  // exact frame index
          
          check4Text.setAutoDraw(true);
        }
        
        
        // if check4Text is active this frame...
        if (check4Text.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *check4ImageLeft* updates
        if (t >= 0.0 && check4ImageLeft.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check4ImageLeft.tStart = t;  // (not accounting for frame time here)
          check4ImageLeft.frameNStart = frameN;  // exact frame index
          
          check4ImageLeft.setAutoDraw(true);
        }
        
        
        // if check4ImageLeft is active this frame...
        if (check4ImageLeft.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *check4ImageRight* updates
        if (t >= 0.0 && check4ImageRight.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check4ImageRight.tStart = t;  // (not accounting for frame time here)
          check4ImageRight.frameNStart = frameN;  // exact frame index
          
          check4ImageRight.setAutoDraw(true);
        }
        
        
        // if check4ImageRight is active this frame...
        if (check4ImageRight.status === PsychoJS.Status.STARTED) {
        }
        
        // Run 'Each Frame' code from check4Select
        if (t_started) {
            if ((t > (t_started + 5))) {
                continueRoutine = false;
            }
        }
        if ((check4Press.keys && (selectionIndicatorPrac.status === NOT_STARTED))) {
            selectionIndicatorPrac.status = PsychoJS.Status.STARTED;
            selectionIndicatorPrac.setPos(selectionPosition(check4Press.keys));
            selectionIndicatorPrac.setAutoDraw(true);
            t_started = new util.Clock().getTime();
        }
        
        
        // *check4Press* updates
        if (t >= 0.0 && check4Press.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check4Press.tStart = t;  // (not accounting for frame time here)
          check4Press.frameNStart = frameN;  // exact frame index
          
          // keyboard checking is just starting
          psychoJS.window.callOnFlip(function() { check4Press.clock.reset(); });  // t=0 on next screen flip
          psychoJS.window.callOnFlip(function() { check4Press.start(); }); // start on screen flip
          psychoJS.window.callOnFlip(function() { check4Press.clearEvents(); });
        }
        
        // if check4Press is active this frame...
        if (check4Press.status === PsychoJS.Status.STARTED) {
          let theseKeys = check4Press.getKeys({keyList: allowedKeys, waitRelease: false});
          _check4Press_allKeys = _check4Press_allKeys.concat(theseKeys);
          if (_check4Press_allKeys.length > 0) {
            check4Press.keys = _check4Press_allKeys[_check4Press_allKeys.length - 1].name;  // just the last key pressed
            check4Press.rt = _check4Press_allKeys[_check4Press_allKeys.length - 1].rt;
            check4Press.duration = _check4Press_allKeys[_check4Press_allKeys.length - 1].duration;
          }
        }
        
        
        // *down4* updates
        if (t >= 0.0 && down4.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          down4.tStart = t;  // (not accounting for frame time here)
          down4.frameNStart = frameN;  // exact frame index
          
          // keyboard checking is just starting
          psychoJS.window.callOnFlip(function() { down4.clock.reset(); });  // t=0 on next screen flip
          psychoJS.window.callOnFlip(function() { down4.start(); }); // start on screen flip
          psychoJS.window.callOnFlip(function() { down4.clearEvents(); });
        }
        
        // if down4 is active this frame...
        if (down4.status === PsychoJS.Status.STARTED) {
          let theseKeys = down4.getKeys({keyList: 'down', waitRelease: false});
          _down4_allKeys = _down4_allKeys.concat(theseKeys);
          if (_down4_allKeys.length > 0) {
            down4.keys = _down4_allKeys[_down4_allKeys.length - 1].name;  // just the last key pressed
            down4.rt = _down4_allKeys[_down4_allKeys.length - 1].rt;
            down4.duration = _down4_allKeys[_down4_allKeys.length - 1].duration;
            // a response ends the routine
            continueRoutine = false;
          }
        }
        
        // check for quit (typically the Esc key)
        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
          return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }
        
        // check if the Routine should terminate
        if (!continueRoutine) {  // a component has requested a forced-end of Routine
          routineForceEnded = true;
          return Scheduler.Event.NEXT;
        }
        
        continueRoutine = false;  // reverts to True if at least one component still running
        for (const thisComponent of check4Components)
          if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
            continueRoutine = true;
            break;
          }
        
        // refresh the screen if continuing
        if (continueRoutine) {
          return Scheduler.Event.FLIP_REPEAT;
        } else {
          return Scheduler.Event.NEXT;
        }
      };
    }
    
    function check4RoutineEnd(snapshot) {
      return async function () {
        //--- Ending Routine 'check4' ---
        for (const thisComponent of check4Components) {
          if (typeof thisComponent.setAutoDraw === 'function') {
            thisComponent.setAutoDraw(false);
          }
        }
        psychoJS.experiment.addData('check4.stopped', globalClock.getTime());
        // Run 'End Routine' code from check4Select
        selectionIndicatorPrac.setAutoDraw(false);
        if ((check4Press.keys === "left")) {
            currentLoop.addData("check4", "correct");
        } else {
            currentLoop.addData("check4", "incorrect");
        }
        
        // update the trial handler
        if (currentLoop instanceof MultiStairHandler) {
          currentLoop.addResponse(check4Press.corr, level);
        }
        psychoJS.experiment.addData('check4Press.keys', check4Press.keys);
        if (typeof check4Press.keys !== 'undefined') {  // we had a response
            psychoJS.experiment.addData('check4Press.rt', check4Press.rt);
            psychoJS.experiment.addData('check4Press.duration', check4Press.duration);
            }
        
        check4Press.stop();
        // update the trial handler
        if (currentLoop instanceof MultiStairHandler) {
          currentLoop.addResponse(down4.corr, level);
        }
        psychoJS.experiment.addData('down4.keys', down4.keys);
        if (typeof down4.keys !== 'undefined') {  // we had a response
            psychoJS.experiment.addData('down4.rt', down4.rt);
            psychoJS.experiment.addData('down4.duration', down4.duration);
            routineTimer.reset();
            }
        
        down4.stop();
        // the Routine "check4" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset();
        
        // Routines running outside a loop should always advance the datafile row
        if (currentLoop === psychoJS.experiment) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        return Scheduler.Event.NEXT;
      }
    }
    
    function check5RoutineBegin(snapshot) {
      return async function () {
        TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
        
        //--- Prepare to start Routine 'check5' ---
        t = 0;
        frameN = -1;
        continueRoutine = true; // until we're told otherwise
        // keep track of whether this Routine was forcibly ended
        routineForceEnded = false;
        check5Clock.reset();
        routineTimer.reset();
        check5MaxDurationReached = false;
        // update component parameters for each repeat
        // Run 'Begin Routine' code from check5Select
        selectionIndicatorPrac.status = NOT_STARTED;
        t_started = false;
        
        check5Press.keys = undefined;
        check5Press.rt = undefined;
        _check5Press_allKeys = [];
        down5.keys = undefined;
        down5.rt = undefined;
        _down5_allKeys = [];
        psychoJS.experiment.addData('check5.started', globalClock.getTime());
        check5MaxDuration = null
        // keep track of which components have finished
        check5Components = [];
        check5Components.push(check5Text);
        check5Components.push(check5ImageLeft);
        check5Components.push(check5ImageRight);
        check5Components.push(check5Press);
        check5Components.push(down5);
        
        for (const thisComponent of check5Components)
          if ('status' in thisComponent)
            thisComponent.status = PsychoJS.Status.NOT_STARTED;
        return Scheduler.Event.NEXT;
      }
    }
    
    function check5RoutineEachFrame() {
      return async function () {
        //--- Loop for each frame of Routine 'check5' ---
        // get current time
        t = check5Clock.getTime();
        frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
        // update/draw components on each frame
        
        // *check5Text* updates
        if (t >= 0.0 && check5Text.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check5Text.tStart = t;  // (not accounting for frame time here)
          check5Text.frameNStart = frameN;  // exact frame index
          
          check5Text.setAutoDraw(true);
        }
        
        
        // if check5Text is active this frame...
        if (check5Text.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *check5ImageLeft* updates
        if (t >= 0.0 && check5ImageLeft.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check5ImageLeft.tStart = t;  // (not accounting for frame time here)
          check5ImageLeft.frameNStart = frameN;  // exact frame index
          
          check5ImageLeft.setAutoDraw(true);
        }
        
        
        // if check5ImageLeft is active this frame...
        if (check5ImageLeft.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *check5ImageRight* updates
        if (t >= 0.0 && check5ImageRight.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check5ImageRight.tStart = t;  // (not accounting for frame time here)
          check5ImageRight.frameNStart = frameN;  // exact frame index
          
          check5ImageRight.setAutoDraw(true);
        }
        
        
        // if check5ImageRight is active this frame...
        if (check5ImageRight.status === PsychoJS.Status.STARTED) {
        }
        
        // Run 'Each Frame' code from check5Select
        if (t_started) {
            if ((t > (t_started + 5))) {
                continueRoutine = false;
            }
        }
        if ((check5Press.keys && (selectionIndicatorPrac.status === NOT_STARTED))) {
            selectionIndicatorPrac.status = PsychoJS.Status.STARTED;
            selectionIndicatorPrac.setPos(selectionPosition(check5Press.keys));
            selectionIndicatorPrac.setAutoDraw(true);
            t_started = new util.Clock().getTime();
        }
        
        
        // *check5Press* updates
        if (t >= 0.0 && check5Press.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          check5Press.tStart = t;  // (not accounting for frame time here)
          check5Press.frameNStart = frameN;  // exact frame index
          
          // keyboard checking is just starting
          psychoJS.window.callOnFlip(function() { check5Press.clock.reset(); });  // t=0 on next screen flip
          psychoJS.window.callOnFlip(function() { check5Press.start(); }); // start on screen flip
          psychoJS.window.callOnFlip(function() { check5Press.clearEvents(); });
        }
        
        // if check5Press is active this frame...
        if (check5Press.status === PsychoJS.Status.STARTED) {
          let theseKeys = check5Press.getKeys({keyList: allowedKeys, waitRelease: false});
          _check5Press_allKeys = _check5Press_allKeys.concat(theseKeys);
          if (_check5Press_allKeys.length > 0) {
            check5Press.keys = _check5Press_allKeys[_check5Press_allKeys.length - 1].name;  // just the last key pressed
            check5Press.rt = _check5Press_allKeys[_check5Press_allKeys.length - 1].rt;
            check5Press.duration = _check5Press_allKeys[_check5Press_allKeys.length - 1].duration;
          }
        }
        
        
        // *down5* updates
        if (t >= 0.0 && down5.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          down5.tStart = t;  // (not accounting for frame time here)
          down5.frameNStart = frameN;  // exact frame index
          
          // keyboard checking is just starting
          psychoJS.window.callOnFlip(function() { down5.clock.reset(); });  // t=0 on next screen flip
          psychoJS.window.callOnFlip(function() { down5.start(); }); // start on screen flip
          psychoJS.window.callOnFlip(function() { down5.clearEvents(); });
        }
        
        // if down5 is active this frame...
        if (down5.status === PsychoJS.Status.STARTED) {
          let theseKeys = down5.getKeys({keyList: 'down', waitRelease: false});
          _down5_allKeys = _down5_allKeys.concat(theseKeys);
          if (_down5_allKeys.length > 0) {
            down5.keys = _down5_allKeys[_down5_allKeys.length - 1].name;  // just the last key pressed
            down5.rt = _down5_allKeys[_down5_allKeys.length - 1].rt;
            down5.duration = _down5_allKeys[_down5_allKeys.length - 1].duration;
            // a response ends the routine
            continueRoutine = false;
          }
        }
        
        // check for quit (typically the Esc key)
        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
          return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }
        
        // check if the Routine should terminate
        if (!continueRoutine) {  // a component has requested a forced-end of Routine
          routineForceEnded = true;
          return Scheduler.Event.NEXT;
        }
        
        continueRoutine = false;  // reverts to True if at least one component still running
        for (const thisComponent of check5Components)
          if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
            continueRoutine = true;
            break;
          }
        
        // refresh the screen if continuing
        if (continueRoutine) {
          return Scheduler.Event.FLIP_REPEAT;
        } else {
          return Scheduler.Event.NEXT;
        }
      };
    }
    
    function check5RoutineEnd(snapshot) {
      return async function () {
        //--- Ending Routine 'check5' ---
        for (const thisComponent of check5Components) {
          if (typeof thisComponent.setAutoDraw === 'function') {
            thisComponent.setAutoDraw(false);
          }
        }
        psychoJS.experiment.addData('check5.stopped', globalClock.getTime());
        // Run 'End Routine' code from check5Select
        selectionIndicatorPrac.setAutoDraw(false);
        if ((check5Press.keys === "left")) {
            currentLoop.addData("check5", "correct");
        } else {
            currentLoop.addData("check5", "incorrect");
        }
        
        // update the trial handler
        if (currentLoop instanceof MultiStairHandler) {
          currentLoop.addResponse(check5Press.corr, level);
        }
        psychoJS.experiment.addData('check5Press.keys', check5Press.keys);
        if (typeof check5Press.keys !== 'undefined') {  // we had a response
            psychoJS.experiment.addData('check5Press.rt', check5Press.rt);
            psychoJS.experiment.addData('check5Press.duration', check5Press.duration);
            }
        
        check5Press.stop();
        // update the trial handler
        if (currentLoop instanceof MultiStairHandler) {
          currentLoop.addResponse(down5.corr, level);
        }
        psychoJS.experiment.addData('down5.keys', down5.keys);
        if (typeof down5.keys !== 'undefined') {  // we had a response
            psychoJS.experiment.addData('down5.rt', down5.rt);
            psychoJS.experiment.addData('down5.duration', down5.duration);
            routineTimer.reset();
            }
        
        down5.stop();
        // the Routine "check5" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset();
        
        // Routines running outside a loop should always advance the datafile row
        if (currentLoop === psychoJS.experiment) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        return Scheduler.Event.NEXT;
      }
    }
    
    function summaryDirectionsRoutineBegin(snapshot) {
      return async function () {
        TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
        
        //--- Prepare to start Routine 'summaryDirections' ---
        t = 0;
        frameN = -1;
        continueRoutine = true; // until we're told otherwise
        // keep track of whether this Routine was forcibly ended
        routineForceEnded = false;
        summaryDirectionsClock.reset();
        routineTimer.reset();
        summaryDirectionsMaxDurationReached = false;
        // update component parameters for each repeat
        advanceScreenPress4.keys = undefined;
        advanceScreenPress4.rt = undefined;
        _advanceScreenPress4_allKeys = [];
        // Run 'Begin Routine' code from summaryLeftCode
        summary.alignText = "left";
        
        psychoJS.experiment.addData('summaryDirections.started', globalClock.getTime());
        summaryDirectionsMaxDuration = null
        // keep track of which components have finished
        summaryDirectionsComponents = [];
        summaryDirectionsComponents.push(summary);
        summaryDirectionsComponents.push(advanceScreenPress4);
        
        for (const thisComponent of summaryDirectionsComponents)
          if ('status' in thisComponent)
            thisComponent.status = PsychoJS.Status.NOT_STARTED;
        return Scheduler.Event.NEXT;
      }
    }
    
    function summaryDirectionsRoutineEachFrame() {
      return async function () {
        //--- Loop for each frame of Routine 'summaryDirections' ---
        // get current time
        t = summaryDirectionsClock.getTime();
        frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
        // update/draw components on each frame
        
        // *summary* updates
        if (t >= 0.0 && summary.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          summary.tStart = t;  // (not accounting for frame time here)
          summary.frameNStart = frameN;  // exact frame index
          
          summary.setAutoDraw(true);
        }
        
        
        // if summary is active this frame...
        if (summary.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *advanceScreenPress4* updates
        if (t >= 0.0 && advanceScreenPress4.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          advanceScreenPress4.tStart = t;  // (not accounting for frame time here)
          advanceScreenPress4.frameNStart = frameN;  // exact frame index
          
          // keyboard checking is just starting
          psychoJS.window.callOnFlip(function() { advanceScreenPress4.clock.reset(); });  // t=0 on next screen flip
          psychoJS.window.callOnFlip(function() { advanceScreenPress4.start(); }); // start on screen flip
          psychoJS.window.callOnFlip(function() { advanceScreenPress4.clearEvents(); });
        }
        
        // if advanceScreenPress4 is active this frame...
        if (advanceScreenPress4.status === PsychoJS.Status.STARTED) {
          let theseKeys = advanceScreenPress4.getKeys({keyList: 'space', waitRelease: false});
          _advanceScreenPress4_allKeys = _advanceScreenPress4_allKeys.concat(theseKeys);
          if (_advanceScreenPress4_allKeys.length > 0) {
            advanceScreenPress4.keys = _advanceScreenPress4_allKeys[_advanceScreenPress4_allKeys.length - 1].name;  // just the last key pressed
            advanceScreenPress4.rt = _advanceScreenPress4_allKeys[_advanceScreenPress4_allKeys.length - 1].rt;
            advanceScreenPress4.duration = _advanceScreenPress4_allKeys[_advanceScreenPress4_allKeys.length - 1].duration;
            // a response ends the routine
            continueRoutine = false;
          }
        }
        
        // check for quit (typically the Esc key)
        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
          return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }
        
        // check if the Routine should terminate
        if (!continueRoutine) {  // a component has requested a forced-end of Routine
          routineForceEnded = true;
          return Scheduler.Event.NEXT;
        }
        
        continueRoutine = false;  // reverts to True if at least one component still running
        for (const thisComponent of summaryDirectionsComponents)
          if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
            continueRoutine = true;
            break;
          }
        
        // refresh the screen if continuing
        if (continueRoutine) {
          return Scheduler.Event.FLIP_REPEAT;
        } else {
          return Scheduler.Event.NEXT;
        }
      };
    }
    
    function summaryDirectionsRoutineEnd(snapshot) {
      return async function () {
        //--- Ending Routine 'summaryDirections' ---
        for (const thisComponent of summaryDirectionsComponents) {
          if (typeof thisComponent.setAutoDraw === 'function') {
            thisComponent.setAutoDraw(false);
          }
        }
        psychoJS.experiment.addData('summaryDirections.stopped', globalClock.getTime());
        // update the trial handler
        if (currentLoop instanceof MultiStairHandler) {
          currentLoop.addResponse(advanceScreenPress4.corr, level);
        }
        psychoJS.experiment.addData('advanceScreenPress4.keys', advanceScreenPress4.keys);
        if (typeof advanceScreenPress4.keys !== 'undefined') {  // we had a response
            psychoJS.experiment.addData('advanceScreenPress4.rt', advanceScreenPress4.rt);
            psychoJS.experiment.addData('advanceScreenPress4.duration', advanceScreenPress4.duration);
            routineTimer.reset();
            }
        
        advanceScreenPress4.stop();
        // the Routine "summaryDirections" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset();
        
        // Routines running outside a loop should always advance the datafile row
        if (currentLoop === psychoJS.experiment) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        return Scheduler.Event.NEXT;
      }
    }
    
    function getReadyRoutineBegin(snapshot) {
      return async function () {
        TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
        
        //--- Prepare to start Routine 'getReady' ---
        t = 0;
        frameN = -1;
        continueRoutine = true; // until we're told otherwise
        // keep track of whether this Routine was forcibly ended
        routineForceEnded = false;
        getReadyClock.reset();
        routineTimer.reset();
        getReadyMaxDurationReached = false;
        // update component parameters for each repeat
        advanceToTrigger.keys = undefined;
        advanceToTrigger.rt = undefined;
        _advanceToTrigger_allKeys = [];
        // Run 'Begin Routine' code from setRunFiles
        console.log(Runs);
        if ((expInfo["startFromRun"] === "1")) {
            if ((Runs === "Run1")) {
                trialOrder = milConditionsFile1;
            }
            if ((Runs === "Run2")) {
                trialOrder = milConditionsFile2;
            }
        } else {
            if ((Runs === "Run1")) {
                trialOrder = milConditionsFile1;
            }
        }
        
        psychoJS.experiment.addData('getReady.started', globalClock.getTime());
        getReadyMaxDuration = null
        // keep track of which components have finished
        getReadyComponents = [];
        getReadyComponents.push(getReadyText);
        getReadyComponents.push(advanceToTrigger);
        
        for (const thisComponent of getReadyComponents)
          if ('status' in thisComponent)
            thisComponent.status = PsychoJS.Status.NOT_STARTED;
        return Scheduler.Event.NEXT;
      }
    }
    
    function getReadyRoutineEachFrame() {
      return async function () {
        //--- Loop for each frame of Routine 'getReady' ---
        // get current time
        t = getReadyClock.getTime();
        frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
        // update/draw components on each frame
        
        // *getReadyText* updates
        if (t >= 0.0 && getReadyText.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          getReadyText.tStart = t;  // (not accounting for frame time here)
          getReadyText.frameNStart = frameN;  // exact frame index
          
          getReadyText.setAutoDraw(true);
        }
        
        
        // if getReadyText is active this frame...
        if (getReadyText.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *advanceToTrigger* updates
        if (t >= 0.0 && advanceToTrigger.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          advanceToTrigger.tStart = t;  // (not accounting for frame time here)
          advanceToTrigger.frameNStart = frameN;  // exact frame index
          
          // keyboard checking is just starting
          psychoJS.window.callOnFlip(function() { advanceToTrigger.clock.reset(); });  // t=0 on next screen flip
          psychoJS.window.callOnFlip(function() { advanceToTrigger.start(); }); // start on screen flip
          psychoJS.window.callOnFlip(function() { advanceToTrigger.clearEvents(); });
        }
        
        // if advanceToTrigger is active this frame...
        if (advanceToTrigger.status === PsychoJS.Status.STARTED) {
          let theseKeys = advanceToTrigger.getKeys({keyList: 'space', waitRelease: false});
          _advanceToTrigger_allKeys = _advanceToTrigger_allKeys.concat(theseKeys);
          if (_advanceToTrigger_allKeys.length > 0) {
            advanceToTrigger.keys = _advanceToTrigger_allKeys[_advanceToTrigger_allKeys.length - 1].name;  // just the last key pressed
            advanceToTrigger.rt = _advanceToTrigger_allKeys[_advanceToTrigger_allKeys.length - 1].rt;
            advanceToTrigger.duration = _advanceToTrigger_allKeys[_advanceToTrigger_allKeys.length - 1].duration;
            // a response ends the routine
            continueRoutine = false;
          }
        }
        
        // check for quit (typically the Esc key)
        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
          return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }
        
        // check if the Routine should terminate
        if (!continueRoutine) {  // a component has requested a forced-end of Routine
          routineForceEnded = true;
          return Scheduler.Event.NEXT;
        }
        
        continueRoutine = false;  // reverts to True if at least one component still running
        for (const thisComponent of getReadyComponents)
          if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
            continueRoutine = true;
            break;
          }
        
        // refresh the screen if continuing
        if (continueRoutine) {
          return Scheduler.Event.FLIP_REPEAT;
        } else {
          return Scheduler.Event.NEXT;
        }
      };
    }
    
    function getReadyRoutineEnd(snapshot) {
      return async function () {
        //--- Ending Routine 'getReady' ---
        for (const thisComponent of getReadyComponents) {
          if (typeof thisComponent.setAutoDraw === 'function') {
            thisComponent.setAutoDraw(false);
          }
        }
        psychoJS.experiment.addData('getReady.stopped', globalClock.getTime());
        // update the trial handler
        if (currentLoop instanceof MultiStairHandler) {
          currentLoop.addResponse(advanceToTrigger.corr, level);
        }
        psychoJS.experiment.addData('advanceToTrigger.keys', advanceToTrigger.keys);
        if (typeof advanceToTrigger.keys !== 'undefined') {  // we had a response
            psychoJS.experiment.addData('advanceToTrigger.rt', advanceToTrigger.rt);
            psychoJS.experiment.addData('advanceToTrigger.duration', advanceToTrigger.duration);
            routineTimer.reset();
            }
        
        advanceToTrigger.stop();
        // the Routine "getReady" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset();
        
        // Routines running outside a loop should always advance the datafile row
        if (currentLoop === psychoJS.experiment) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        return Scheduler.Event.NEXT;
      }
    }
    
    function waitForScannerRoutineBegin(snapshot) {
      return async function () {
        TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
        
        //--- Prepare to start Routine 'waitForScanner' ---
        t = 0;
        frameN = -1;
        continueRoutine = true; // until we're told otherwise
        // keep track of whether this Routine was forcibly ended
        routineForceEnded = false;
        waitForScannerClock.reset();
        routineTimer.reset();
        waitForScannerMaxDurationReached = false;
        // update component parameters for each repeat
        // Run 'Begin Routine' code from waitScannerCode
        if ((expInfo["mode"] !== "scan")) {
            continueRoutine = false;
        }
        /*
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
        */
        
        scannerTriggerKey.keys = undefined;
        scannerTriggerKey.rt = undefined;
        _scannerTriggerKey_allKeys = [];
        // Run 'Begin Routine' code from setImg
        hg1Img = ("stimuli_sets/" + cuePairs["highgain"]["optimalChoice"]);
        hg2Img = ("stimuli_sets/" + cuePairs["highgain"]["suboptimalChoice"]);
        lg1Img = ("stimuli_sets/" + cuePairs["lowgain"]["suboptimalChoice"]);
        lg2Img = ("stimuli_sets/" + cuePairs["lowgain"]["optimalChoice"]);
        hl1Img = ("stimuli_sets/" + cuePairs["highloss"]["suboptimalChoice"]);
        hl2Img = ("stimuli_sets/" + cuePairs["highloss"]["optimalChoice"]);
        ll1Img = ("stimuli_sets/" + cuePairs["lowloss"]["optimalChoice"]);
        ll2Img = ("stimuli_sets/" + cuePairs["lowloss"]["suboptimalChoice"]);
        
        hg1.setImage(hg1Img);
        hg2.setImage(hg2Img);
        lg1.setImage(lg1Img);
        lg2.setImage(lg2Img);
        hl1.setImage(hl1Img);
        hl2.setImage(hl2Img);
        ll1.setImage(ll1Img);
        ll2.setImage(ll2Img);
        psychoJS.experiment.addData('waitForScanner.started', globalClock.getTime());
        waitForScannerMaxDuration = null
        // keep track of which components have finished
        waitForScannerComponents = [];
        waitForScannerComponents.push(scannerTriggerKey);
        waitForScannerComponents.push(sumTemplate);
        waitForScannerComponents.push(hg1);
        waitForScannerComponents.push(hg2);
        waitForScannerComponents.push(lg1);
        waitForScannerComponents.push(lg2);
        waitForScannerComponents.push(hl1);
        waitForScannerComponents.push(hl2);
        waitForScannerComponents.push(ll1);
        waitForScannerComponents.push(ll2);
        
        for (const thisComponent of waitForScannerComponents)
          if ('status' in thisComponent)
            thisComponent.status = PsychoJS.Status.NOT_STARTED;
        return Scheduler.Event.NEXT;
      }
    }
    
    function waitForScannerRoutineEachFrame() {
      return async function () {
        //--- Loop for each frame of Routine 'waitForScanner' ---
        // get current time
        t = waitForScannerClock.getTime();
        frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
        // update/draw components on each frame
        
        // *scannerTriggerKey* updates
        if (t >= 0.0 && scannerTriggerKey.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          scannerTriggerKey.tStart = t;  // (not accounting for frame time here)
          scannerTriggerKey.frameNStart = frameN;  // exact frame index
          
          // keyboard checking is just starting
          psychoJS.window.callOnFlip(function() { scannerTriggerKey.clock.reset(); });  // t=0 on next screen flip
          psychoJS.window.callOnFlip(function() { scannerTriggerKey.start(); }); // start on screen flip
          psychoJS.window.callOnFlip(function() { scannerTriggerKey.clearEvents(); });
        }
        
        // if scannerTriggerKey is active this frame...
        if (scannerTriggerKey.status === PsychoJS.Status.STARTED) {
          let theseKeys = scannerTriggerKey.getKeys({keyList: 't', waitRelease: false});
          _scannerTriggerKey_allKeys = _scannerTriggerKey_allKeys.concat(theseKeys);
          if (_scannerTriggerKey_allKeys.length > 0) {
            scannerTriggerKey.keys = _scannerTriggerKey_allKeys[_scannerTriggerKey_allKeys.length - 1].name;  // just the last key pressed
            scannerTriggerKey.rt = _scannerTriggerKey_allKeys[_scannerTriggerKey_allKeys.length - 1].rt;
            scannerTriggerKey.duration = _scannerTriggerKey_allKeys[_scannerTriggerKey_allKeys.length - 1].duration;
            // a response ends the routine
            continueRoutine = false;
          }
        }
        
        
        // *sumTemplate* updates
        if (t >= 0.0 && sumTemplate.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          sumTemplate.tStart = t;  // (not accounting for frame time here)
          sumTemplate.frameNStart = frameN;  // exact frame index
          
          sumTemplate.setAutoDraw(true);
        }
        
        
        // if sumTemplate is active this frame...
        if (sumTemplate.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *hg1* updates
        if (t >= 0.0 && hg1.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          hg1.tStart = t;  // (not accounting for frame time here)
          hg1.frameNStart = frameN;  // exact frame index
          
          hg1.setAutoDraw(true);
        }
        
        
        // if hg1 is active this frame...
        if (hg1.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *hg2* updates
        if (t >= 0.0 && hg2.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          hg2.tStart = t;  // (not accounting for frame time here)
          hg2.frameNStart = frameN;  // exact frame index
          
          hg2.setAutoDraw(true);
        }
        
        
        // if hg2 is active this frame...
        if (hg2.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *lg1* updates
        if (t >= 0.0 && lg1.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          lg1.tStart = t;  // (not accounting for frame time here)
          lg1.frameNStart = frameN;  // exact frame index
          
          lg1.setAutoDraw(true);
        }
        
        
        // if lg1 is active this frame...
        if (lg1.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *lg2* updates
        if (t >= 0.0 && lg2.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          lg2.tStart = t;  // (not accounting for frame time here)
          lg2.frameNStart = frameN;  // exact frame index
          
          lg2.setAutoDraw(true);
        }
        
        
        // if lg2 is active this frame...
        if (lg2.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *hl1* updates
        if (t >= 0.0 && hl1.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          hl1.tStart = t;  // (not accounting for frame time here)
          hl1.frameNStart = frameN;  // exact frame index
          
          hl1.setAutoDraw(true);
        }
        
        
        // if hl1 is active this frame...
        if (hl1.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *hl2* updates
        if (t >= 0.0 && hl2.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          hl2.tStart = t;  // (not accounting for frame time here)
          hl2.frameNStart = frameN;  // exact frame index
          
          hl2.setAutoDraw(true);
        }
        
        
        // if hl2 is active this frame...
        if (hl2.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *ll1* updates
        if (t >= 0.0 && ll1.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          ll1.tStart = t;  // (not accounting for frame time here)
          ll1.frameNStart = frameN;  // exact frame index
          
          ll1.setAutoDraw(true);
        }
        
        
        // if ll1 is active this frame...
        if (ll1.status === PsychoJS.Status.STARTED) {
        }
        
        
        // *ll2* updates
        if (t >= 0.0 && ll2.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          ll2.tStart = t;  // (not accounting for frame time here)
          ll2.frameNStart = frameN;  // exact frame index
          
          ll2.setAutoDraw(true);
        }
        
        
        // if ll2 is active this frame...
        if (ll2.status === PsychoJS.Status.STARTED) {
        }
        
        // check for quit (typically the Esc key)
        if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
          return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
        }
        
        // check if the Routine should terminate
        if (!continueRoutine) {  // a component has requested a forced-end of Routine
          routineForceEnded = true;
          return Scheduler.Event.NEXT;
        }
        
        continueRoutine = false;  // reverts to True if at least one component still running
        for (const thisComponent of waitForScannerComponents)
          if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
            continueRoutine = true;
            break;
          }
        
        // refresh the screen if continuing
        if (continueRoutine) {
          return Scheduler.Event.FLIP_REPEAT;
        } else {
          return Scheduler.Event.NEXT;
        }
      };
    }
    
    function waitForScannerRoutineEnd(snapshot) {
      return async function () {
        //--- Ending Routine 'waitForScanner' ---
        for (const thisComponent of waitForScannerComponents) {
          if (typeof thisComponent.setAutoDraw === 'function') {
            thisComponent.setAutoDraw(false);
          }
        }
        psychoJS.experiment.addData('waitForScanner.stopped', globalClock.getTime());
        // Run 'End Routine' code from waitScannerCode
        fmriClock.reset();
        
        // update the trial handler
        if (currentLoop instanceof MultiStairHandler) {
          currentLoop.addResponse(scannerTriggerKey.corr, level);
        }
        psychoJS.experiment.addData('scannerTriggerKey.keys', scannerTriggerKey.keys);
        if (typeof scannerTriggerKey.keys !== 'undefined') {  // we had a response
            psychoJS.experiment.addData('scannerTriggerKey.rt', scannerTriggerKey.rt);
            psychoJS.experiment.addData('scannerTriggerKey.duration', scannerTriggerKey.duration);
            routineTimer.reset();
            }
        
        scannerTriggerKey.stop();
        // Run 'End Routine' code from frameOnCode
        frame.setAutoDraw(true);
        
        // the Routine "waitForScanner" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset();
        
        // Routines running outside a loop should always advance the datafile row
        if (currentLoop === psychoJS.experiment) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        return Scheduler.Event.NEXT;
      }
    }
    
    function cueRoutineBegin(snapshot) {
      return async function () {
        TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
        
        //--- Prepare to start Routine 'cue' ---
        t = 0;
        frameN = -1;
        continueRoutine = true; // until we're told otherwise
        // keep track of whether this Routine was forcibly ended
        routineForceEnded = false;
        cueClock.reset(routineTimer.getTime());
        routineTimer.add(3.000000);
        cueMaxDurationReached = false;
        // update component parameters for each repeat
        cueResp.keys = undefined;
        cueResp.rt = undefined;
        _cueResp_allKeys = [];
        // Run 'Begin Routine' code from setCues
        /* Syntax Error: Fix Python code */
        // Run 'Begin Routine' code from frameLabels
        if ((condition === "lowgain")) {
            corrvalue = 0.2;
            incorrvalue = 0;
            topLabel = `+$${Math.abs(corrvalue)}`;
            bottomLabel = `+$${Math.abs(incorrvalue)}`;
        }
        if ((condition === "highgain")) {
            corrvalue = 1.0;
            incorrvalue = 0;
            topLabel = `+$${Math.abs(corrvalue)}`;
            bottomLabel = `+$${Math.abs(incorrvalue)}`;
        }
        if ((condition === "lowloss")) {
            corrvalue = 0;
            incorrvalue = (- 0.2);
            topLabel = `-$${Math.abs(corrvalue)}`;
            bottomLabel = `-$${Math.abs(incorrvalue)}`;
        }
        if ((condition === "highloss")) {
            corrvalue = 0;
            incorrvalue = (- 1.0);
            topLabel = `-$${Math.abs(corrvalue)}`;
            bottomLabel = `-$${Math.abs(incorrvalue)}`;
        }
        topFrameText.setText(topLabel);
        topFrameText.status = NOT_STARTED;
        bottomFrameText.setText(bottomLabel);
        bottomFrameText.status = NOT_STARTED;
        
        // Run 'Begin Routine' code from setImageTiming
        cueTimeAdded = false;
        responseTimeAdded = false;
        feedbackTimeAdded = false;
        currentLoop.addData("startFixOnTime", fmriClock.getTime());
        
        // Run 'Begin Routine' code from manageTrialsLoop
        trialsClock.reset();
        
        // Run 'Begin Routine' code from selectionIndicatorCode
        cueEndTime = 3;
        selectionIndicator.status = NOT_STARTED;
        
        psychoJS.experiment.addData('cue.started', globalClock.getTime());
        cueMaxDuration = null
        // keep track of which components have finished
        cueComponents = [];
        cueComponents.push(startFixStatic);
        cueComponents.push(leftCue);
        cueComponents.push(rightCue);
        cueComponents.push(cueResp);
        cueComponents.push(staticISI);
        
        for (const thisComponent of cueComponents)
          if ('status' in thisComponent)
            thisComponent.status = PsychoJS.Status.NOT_STARTED;
        return Scheduler.Event.NEXT;
      }
    }
    
    function cueRoutineEachFrame() {
      return async function () {
        //--- Loop for each frame of Routine 'cue' ---
        // get current time
        t = cueClock.getTime();
        frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
        // update/draw components on each frame
        
        // *startFixStatic* updates
        if (t >= 0.0 && startFixStatic.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          startFixStatic.tStart = t;  // (not accounting for frame time here)
          startFixStatic.frameNStart = frameN;  // exact frame index
          
          startFixStatic.setAutoDraw(true);
        }
        
        
        // if startFixStatic is active this frame...
        if (startFixStatic.status === PsychoJS.Status.STARTED) {
        }
        
        frameRemains = 0.0 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
        if (startFixStatic.status === PsychoJS.Status.STARTED && t >= frameRemains) {
          // keep track of stop time/frame for later
          startFixStatic.tStop = t;  // not accounting for scr refresh
          startFixStatic.frameNStop = frameN;  // exact frame index
          // update status
          startFixStatic.status = PsychoJS.Status.FINISHED;
          startFixStatic.setAutoDraw(false);
        }
        
        
        // *leftCue* updates
        if (t >= 0.5 && leftCue.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          leftCue.tStart = t;  // (not accounting for frame time here)
          leftCue.frameNStart = frameN;  // exact frame index
          
          leftCue.setAutoDraw(true);
        }
        
        
        // if leftCue is active this frame...
        if (leftCue.status === PsychoJS.Status.STARTED) {
        }
        
        frameRemains = 0.5 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
        if (leftCue.status === PsychoJS.Status.STARTED && t >= frameRemains) {
          // keep track of stop time/frame for later
          leftCue.tStop = t;  // not accounting for scr refresh
          leftCue.frameNStop = frameN;  // exact frame index
          // update status
          leftCue.status = PsychoJS.Status.FINISHED;
          leftCue.setAutoDraw(false);
        }
        
        
        // *rightCue* updates
        if (t >= 0.5 && rightCue.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          rightCue.tStart = t;  // (not accounting for frame time here)
          rightCue.frameNStart = frameN;  // exact frame index
          
          rightCue.setAutoDraw(true);
        }
        
        
        // if rightCue is active this frame...
        if (rightCue.status === PsychoJS.Status.STARTED) {
        }
        
        frameRemains = 0.5 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
        if (rightCue.status === PsychoJS.Status.STARTED && t >= frameRemains) {
          // keep track of stop time/frame for later
          rightCue.tStop = t;  // not accounting for scr refresh
          rightCue.frameNStop = frameN;  // exact frame index
          // update status
          rightCue.status = PsychoJS.Status.FINISHED;
          rightCue.setAutoDraw(false);
        }
        
        
        // *cueResp* updates
        if (t >= 0.5 && cueResp.status === PsychoJS.Status.NOT_STARTED) {
          // keep track of start time/frame for later
          cueResp.tStart = t;  // (not accounting for frame time here)
          cueResp.frameNStart = frameN;  // exact frame index
          
          // keyboard checking is just starting
          psychoJS.window.callOnFlip(function() { cueResp.clock.reset(); });  // t=0 on next screen flip
          psychoJS.window.callOnFlip(function() { cueResp.start(); }); // start on screen flip
          psychoJS.window.callOnFlip(function() { cueResp.clearEvents(); });
        }
        frameRemains = 0.5 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
        if (cueResp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
          // keep track of stop time/frame for later
          cueResp.tStop = t;  // not accounting for scr refresh
          cueResp.frameNStop = frameN;  // exact frame index
          // update status
          cueResp.status = PsychoJS.Status.FINISHED;
          frameRemains = 0.5 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
          if (cueResp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
            // keep track of stop time/frame for later
            cueResp.tStop = t;  // not accounting for scr refresh
            cueResp.frameNStop = frameN;  // exact frame index
            // update status
            cueResp.status = PsychoJS.Status.FINISHED;
            cueResp.status = PsychoJS.Status.FINISHED;
              }
            
          }
          
          // if cueResp is active this frame...
          if (cueResp.status === PsychoJS.Status.STARTED) {
            let theseKeys = cueResp.getKeys({keyList: allowedKeys, waitRelease: false});
            _cueResp_allKeys = _cueResp_allKeys.concat(theseKeys);
            if (_cueResp_allKeys.length > 0) {
              cueResp.keys = _cueResp_allKeys[0].name;  // just the first key pressed
              cueResp.rt = _cueResp_allKeys[0].rt;
              cueResp.duration = _cueResp_allKeys[0].duration;
              // was this correct?
              if (cueResp.keys == correct) {
                  cueResp.corr = 1;
              } else {
                  cueResp.corr = 0;
              }
            }
          }
          
          // Run 'Each Frame' code from frameLabels
          if (((t > 0.5) && (topFrameText.status === NOT_STARTED))) {
              topFrameText.setAutoDraw(true);
              bottomFrameText.setAutoDraw(true);
              topFrameText.status = PsychoJS.Status.STARTED;
          }
          
          // Run 'Each Frame' code from setImageTiming
          if ((! cueTimeAdded)) {
              if ((leftCue.autoDraw === true)) {
                  currentLoop.addData("cueOnTime", fmriClock.getTime());
                  cueTimeAdded = true;
              }
          }
          if (((! responseTimeAdded) && cueResp.keys.length)) {
              currentLoop.addData("responseTime", fmriClock.getTime());
              responseTimeAdded = true;
          }
          
          // Run 'Each Frame' code from selectionIndicatorCode
          if ((t > cueEndTime)) {
              continueRoutine = false;
          }
          if ((cueResp.keys && (selectionIndicator.status === NOT_STARTED))) {
              cueComponents.push(selectionIndicator);
              selectionIndicator.status = PsychoJS.Status.STARTED;
              selectionIndicator.setPos(selectionPosition(cueResp.keys));
              selectionIndicator.setAutoDraw(true);
          }
          
          if (t >= 0.0 && staticISI.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            staticISI.tStart = t;  // (not accounting for frame time here)
            staticISI.frameNStart = frameN;  // exact frame index
            
            staticISI.status = PsychoJS.Status.STARTED;
            // Updating other components during *staticISI*
            console.log('register and start downloading resources specified by component leftCue');
            await psychoJS.serverManager.prepareResources(leftImage);
            staticISI.status = PsychoJS.Status.STARTED;
            leftCue.setImage(leftImage);
            console.log('register and start downloading resources specified by component rightCue');
            await psychoJS.serverManager.prepareResources(rightImage);
            staticISI.status = PsychoJS.Status.STARTED;
            rightCue.setImage(rightImage);
            // Component updates done
          }
          frameRemains = 0.0 + 0.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
          if (staticISI.status === PsychoJS.Status.STARTED && t >= frameRemains) {
            // keep track of stop time/frame for later
            staticISI.tStop = t;  // not accounting for scr refresh
            staticISI.frameNStop = frameN;  // exact frame index
            // update status
            staticISI.status = PsychoJS.Status.FINISHED;
            if (psychoJS.serverManager.getResourceStatus(leftImage) === core.ServerManager.ResourceStatus.DOWNLOADED) {
              console.log('finished downloading resources specified by component staticISI');
            } else {
              console.log('resource specified in staticISI took longer than expected to download');
              await waitForResources(resources = leftImage)
            }
            if (psychoJS.serverManager.getResourceStatus(rightImage) === core.ServerManager.ResourceStatus.DOWNLOADED) {
              console.log('finished downloading resources specified by component staticISI');
            } else {
              console.log('resource specified in staticISI took longer than expected to download');
              await waitForResources(resources = rightImage)
            }
            staticISI.status = PsychoJS.Status.FINISHED;
          }
          // check for quit (typically the Esc key)
          if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
          }
          
          // check if the Routine should terminate
          if (!continueRoutine) {  // a component has requested a forced-end of Routine
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
          }
          
          continueRoutine = false;  // reverts to True if at least one component still running
          for (const thisComponent of cueComponents)
            if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
              continueRoutine = true;
              break;
            }
          
          // refresh the screen if continuing
          if (continueRoutine && routineTimer.getTime() > 0) {
            return Scheduler.Event.FLIP_REPEAT;
          } else {
            return Scheduler.Event.NEXT;
          }
        };
      }
      
      function cueRoutineEnd(snapshot) {
        return async function () {
          //--- Ending Routine 'cue' ---
          for (const thisComponent of cueComponents) {
            if (typeof thisComponent.setAutoDraw === 'function') {
              thisComponent.setAutoDraw(false);
            }
          }
          psychoJS.experiment.addData('cue.stopped', globalClock.getTime());
          // was no response the correct answer?!
          if (cueResp.keys === undefined) {
            if (['None','none',undefined].includes(correct)) {
               cueResp.corr = 1;  // correct non-response
            } else {
               cueResp.corr = 0;  // failed to respond (incorrectly)
            }
          }
          // store data for current loop
          // update the trial handler
          if (currentLoop instanceof MultiStairHandler) {
            currentLoop.addResponse(cueResp.corr, level);
          }
          psychoJS.experiment.addData('cueResp.keys', cueResp.keys);
          psychoJS.experiment.addData('cueResp.corr', cueResp.corr);
          if (typeof cueResp.keys !== 'undefined') {  // we had a response
              psychoJS.experiment.addData('cueResp.rt', cueResp.rt);
              psychoJS.experiment.addData('cueResp.duration', cueResp.duration);
              }
          
          cueResp.stop();
          // Run 'End Routine' code from setCues
          if ((cueResp.keys === responseKeys["left"])) {
              currentLoop.addData("optedSide", "left");
              currentLoop.addData("optedImg", leftImage);
          } else {
              if ((cueResp.keys === responseKeys["right"])) {
                  currentLoop.addData("optedSide", "right");
                  currentLoop.addData("optedImg", rightImage);
              } else {
                  currentLoop.addData("optedSide", "NA");
                  currentLoop.addData("optedImg", "NA");
              }
          }
          
          // Run 'End Routine' code from frameLabels
          topFrameText.status = NOT_STARTED;
          
          // Run 'End Routine' code from setImageTiming
          currentLoop.addData("cueOffTime", fmriClock.getTime());
          
          if (routineForceEnded) {
              routineTimer.reset();} else if (cueMaxDurationReached) {
              cueClock.add(cueMaxDuration);
          } else {
              cueClock.add(3.000000);
          }
          // Routines running outside a loop should always advance the datafile row
          if (currentLoop === psychoJS.experiment) {
            psychoJS.experiment.nextEntry(snapshot);
          }
          return Scheduler.Event.NEXT;
        }
      }
      
      function outcomeDelayRoutineRoutineBegin(snapshot) {
        return async function () {
          TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
          
          //--- Prepare to start Routine 'outcomeDelayRoutine' ---
          t = 0;
          frameN = -1;
          continueRoutine = true; // until we're told otherwise
          // keep track of whether this Routine was forcibly ended
          routineForceEnded = false;
          outcomeDelayRoutineClock.reset();
          routineTimer.reset();
          outcomeDelayRoutineMaxDurationReached = false;
          // update component parameters for each repeat
          psychoJS.experiment.addData('outcomeDelayRoutine.started', globalClock.getTime());
          outcomeDelayRoutineMaxDuration = null
          // keep track of which components have finished
          outcomeDelayRoutineComponents = [];
          outcomeDelayRoutineComponents.push(outcomeDelayFix);
          
          for (const thisComponent of outcomeDelayRoutineComponents)
            if ('status' in thisComponent)
              thisComponent.status = PsychoJS.Status.NOT_STARTED;
          return Scheduler.Event.NEXT;
        }
      }
      
      function outcomeDelayRoutineRoutineEachFrame() {
        return async function () {
          //--- Loop for each frame of Routine 'outcomeDelayRoutine' ---
          // get current time
          t = outcomeDelayRoutineClock.getTime();
          frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
          // update/draw components on each frame
          
          // *outcomeDelayFix* updates
          if (t >= 0.0 && outcomeDelayFix.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            outcomeDelayFix.tStart = t;  // (not accounting for frame time here)
            outcomeDelayFix.frameNStart = frameN;  // exact frame index
            
            outcomeDelayFix.setAutoDraw(true);
          }
          
          
          // if outcomeDelayFix is active this frame...
          if (outcomeDelayFix.status === PsychoJS.Status.STARTED) {
          }
          
          frameRemains = 0.0 + outcomeDelay - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
          if (outcomeDelayFix.status === PsychoJS.Status.STARTED && t >= frameRemains) {
            // keep track of stop time/frame for later
            outcomeDelayFix.tStop = t;  // not accounting for scr refresh
            outcomeDelayFix.frameNStop = frameN;  // exact frame index
            // update status
            outcomeDelayFix.status = PsychoJS.Status.FINISHED;
            outcomeDelayFix.setAutoDraw(false);
          }
          
          // check for quit (typically the Esc key)
          if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
          }
          
          // check if the Routine should terminate
          if (!continueRoutine) {  // a component has requested a forced-end of Routine
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
          }
          
          continueRoutine = false;  // reverts to True if at least one component still running
          for (const thisComponent of outcomeDelayRoutineComponents)
            if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
              continueRoutine = true;
              break;
            }
          
          // refresh the screen if continuing
          if (continueRoutine) {
            return Scheduler.Event.FLIP_REPEAT;
          } else {
            return Scheduler.Event.NEXT;
          }
        };
      }
      
      function outcomeDelayRoutineRoutineEnd(snapshot) {
        return async function () {
          //--- Ending Routine 'outcomeDelayRoutine' ---
          for (const thisComponent of outcomeDelayRoutineComponents) {
            if (typeof thisComponent.setAutoDraw === 'function') {
              thisComponent.setAutoDraw(false);
            }
          }
          psychoJS.experiment.addData('outcomeDelayRoutine.stopped', globalClock.getTime());
          // the Routine "outcomeDelayRoutine" was not non-slip safe, so reset the non-slip timer
          routineTimer.reset();
          
          // Routines running outside a loop should always advance the datafile row
          if (currentLoop === psychoJS.experiment) {
            psychoJS.experiment.nextEntry(snapshot);
          }
          return Scheduler.Event.NEXT;
        }
      }
      
      function outcomeRoutineRoutineBegin(snapshot) {
        return async function () {
          TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
          
          //--- Prepare to start Routine 'outcomeRoutine' ---
          t = 0;
          frameN = -1;
          continueRoutine = true; // until we're told otherwise
          // keep track of whether this Routine was forcibly ended
          routineForceEnded = false;
          outcomeRoutineClock.reset(routineTimer.getTime());
          routineTimer.add(1.000000);
          outcomeRoutineMaxDurationReached = false;
          // update component parameters for each repeat
          // Run 'Begin Routine' code from rightChoice
          if ((cueResp.keys === sidePositions[optimalImg])) {
              Choice = "optimal";
          } else {
              if ((cueResp.keys === sidePositions[suboptimalImg])) {
                  Choice = "suboptimal";
              } else {
                  Choice = "NA";
              }
          }
          Accuracy = cueResp.corr;
          currentLoop.addData("optedFor", Choice);
          currentLoop.addData("accuracy", Accuracy);
          
          // Run 'Begin Routine' code from defineOutcomes
          if ((condition === "highgain")) {
              trueOperator = "+";
              falseOperator = "+";
          } else {
              if ((condition === "lowgain")) {
                  trueOperator = "+";
                  falseOperator = "+";
              } else {
                  if ((condition === "highloss")) {
                      trueOperator = "-";
                      falseOperator = "-";
                  } else {
                      trueOperator = "-";
                      falseOperator = "-";
                  }
              }
          }
          if (cueResp.keys) {
              outcome = choose_outcome_from_cue(cueResp);
              currentLoop.addData("outcome", outcome);
              if (outcome) {
                  change = corrvalue;
                  outcome_msg = `${trueOperator}$${Math.abs(change)}`;
                  outcome_image = null;
                  Feedback = 1;
              } else {
                  change = incorrvalue;
                  outcome_msg = `${falseOperator}$${Math.abs(change)}`;
                  outcome_image = null;
                  Feedback = 0;
              }
          } else {
              change = incorrvalue;
              currentLoop.addData("outcome", null);
              outcome_msg = `${falseOperator}$${Math.abs(change)}`;
              outcome_image = null;
              Feedback = "NA";
          }
          bank += change;
          currentLoop.addData("Feedback", Feedback);
          currentLoop.addData("gainLossAmount", change);
          currentLoop.addData("runningBankTotal", bank);
          currentLoop.addData("feedbackTime", fmriClock.getTime());
          
          outcomeText.setText(outcome_msg);
          psychoJS.experiment.addData('outcomeRoutine.started', globalClock.getTime());
          outcomeRoutineMaxDuration = null
          // keep track of which components have finished
          outcomeRoutineComponents = [];
          outcomeRoutineComponents.push(outcomeText);
          
          for (const thisComponent of outcomeRoutineComponents)
            if ('status' in thisComponent)
              thisComponent.status = PsychoJS.Status.NOT_STARTED;
          return Scheduler.Event.NEXT;
        }
      }
      
      function outcomeRoutineRoutineEachFrame() {
        return async function () {
          //--- Loop for each frame of Routine 'outcomeRoutine' ---
          // get current time
          t = outcomeRoutineClock.getTime();
          frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
          // update/draw components on each frame
          
          // *outcomeText* updates
          if (t >= 0.0 && outcomeText.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            outcomeText.tStart = t;  // (not accounting for frame time here)
            outcomeText.frameNStart = frameN;  // exact frame index
            
            outcomeText.setAutoDraw(true);
          }
          
          
          // if outcomeText is active this frame...
          if (outcomeText.status === PsychoJS.Status.STARTED) {
          }
          
          frameRemains = 0.0 + 1.0 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
          if (outcomeText.status === PsychoJS.Status.STARTED && t >= frameRemains) {
            // keep track of stop time/frame for later
            outcomeText.tStop = t;  // not accounting for scr refresh
            outcomeText.frameNStop = frameN;  // exact frame index
            // update status
            outcomeText.status = PsychoJS.Status.FINISHED;
            outcomeText.setAutoDraw(false);
          }
          
          // check for quit (typically the Esc key)
          if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
          }
          
          // check if the Routine should terminate
          if (!continueRoutine) {  // a component has requested a forced-end of Routine
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
          }
          
          continueRoutine = false;  // reverts to True if at least one component still running
          for (const thisComponent of outcomeRoutineComponents)
            if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
              continueRoutine = true;
              break;
            }
          
          // refresh the screen if continuing
          if (continueRoutine && routineTimer.getTime() > 0) {
            return Scheduler.Event.FLIP_REPEAT;
          } else {
            return Scheduler.Event.NEXT;
          }
        };
      }
      
      function outcomeRoutineRoutineEnd(snapshot) {
        return async function () {
          //--- Ending Routine 'outcomeRoutine' ---
          for (const thisComponent of outcomeRoutineComponents) {
            if (typeof thisComponent.setAutoDraw === 'function') {
              thisComponent.setAutoDraw(false);
            }
          }
          psychoJS.experiment.addData('outcomeRoutine.stopped', globalClock.getTime());
          // Run 'End Routine' code from defineOutcomes
          expInfo["bank"] = bank;
          currentLoop.addData("feedbackOffTime", fmriClock.getTime());
          
          // Run 'End Routine' code from frameLabelsOff
          topFrameText.setAutoDraw(false);
          bottomFrameText.setAutoDraw(false);
          
          if (routineForceEnded) {
              routineTimer.reset();} else if (outcomeRoutineMaxDurationReached) {
              outcomeRoutineClock.add(outcomeRoutineMaxDuration);
          } else {
              outcomeRoutineClock.add(1.000000);
          }
          // Routines running outside a loop should always advance the datafile row
          if (currentLoop === psychoJS.experiment) {
            psychoJS.experiment.nextEntry(snapshot);
          }
          return Scheduler.Event.NEXT;
        }
      }
      
      function fixationRoutineBegin(snapshot) {
        return async function () {
          TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
          
          //--- Prepare to start Routine 'fixation' ---
          t = 0;
          frameN = -1;
          continueRoutine = true; // until we're told otherwise
          // keep track of whether this Routine was forcibly ended
          routineForceEnded = false;
          fixationClock.reset();
          routineTimer.reset();
          fixationMaxDurationReached = false;
          // update component parameters for each repeat
          // Run 'Begin Routine' code from loopDurationCode
          currentLoop.addData("fixationTime", fmriClock.getTime());
          
          psychoJS.experiment.addData('fixation.started', globalClock.getTime());
          fixationMaxDuration = null
          // keep track of which components have finished
          fixationComponents = [];
          fixationComponents.push(fixationCrosshair);
          
          for (const thisComponent of fixationComponents)
            if ('status' in thisComponent)
              thisComponent.status = PsychoJS.Status.NOT_STARTED;
          return Scheduler.Event.NEXT;
        }
      }
      
      function fixationRoutineEachFrame() {
        return async function () {
          //--- Loop for each frame of Routine 'fixation' ---
          // get current time
          t = fixationClock.getTime();
          frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
          // update/draw components on each frame
          
          // *fixationCrosshair* updates
          if (t >= 0.0 && fixationCrosshair.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            fixationCrosshair.tStart = t;  // (not accounting for frame time here)
            fixationCrosshair.frameNStart = frameN;  // exact frame index
            
            fixationCrosshair.setAutoDraw(true);
          }
          
          
          // if fixationCrosshair is active this frame...
          if (fixationCrosshair.status === PsychoJS.Status.STARTED) {
          }
          
          frameRemains = 0.0 + fixationDuration - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
          if (fixationCrosshair.status === PsychoJS.Status.STARTED && t >= frameRemains) {
            // keep track of stop time/frame for later
            fixationCrosshair.tStop = t;  // not accounting for scr refresh
            fixationCrosshair.frameNStop = frameN;  // exact frame index
            // update status
            fixationCrosshair.status = PsychoJS.Status.FINISHED;
            fixationCrosshair.setAutoDraw(false);
          }
          
          // check for quit (typically the Esc key)
          if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
          }
          
          // check if the Routine should terminate
          if (!continueRoutine) {  // a component has requested a forced-end of Routine
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
          }
          
          continueRoutine = false;  // reverts to True if at least one component still running
          for (const thisComponent of fixationComponents)
            if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
              continueRoutine = true;
              break;
            }
          
          // refresh the screen if continuing
          if (continueRoutine) {
            return Scheduler.Event.FLIP_REPEAT;
          } else {
            return Scheduler.Event.NEXT;
          }
        };
      }
      
      function fixationRoutineEnd(snapshot) {
        return async function () {
          //--- Ending Routine 'fixation' ---
          for (const thisComponent of fixationComponents) {
            if (typeof thisComponent.setAutoDraw === 'function') {
              thisComponent.setAutoDraw(false);
            }
          }
          psychoJS.experiment.addData('fixation.stopped', globalClock.getTime());
          // Run 'End Routine' code from loopDurationCode
          logging.exp(`Measured Fixation Dureation: ${t}`);
          logging.exp(`Trials Clock: ${trialsClock.getTime()}`);
          
          // the Routine "fixation" was not non-slip safe, so reset the non-slip timer
          routineTimer.reset();
          
          // Routines running outside a loop should always advance the datafile row
          if (currentLoop === psychoJS.experiment) {
            psychoJS.experiment.nextEntry(snapshot);
          }
          return Scheduler.Event.NEXT;
        }
      }
      
      function breakScreenRoutineBegin(snapshot) {
        return async function () {
          TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
          
          //--- Prepare to start Routine 'breakScreen' ---
          t = 0;
          frameN = -1;
          continueRoutine = true; // until we're told otherwise
          // keep track of whether this Routine was forcibly ended
          routineForceEnded = false;
          breakScreenClock.reset();
          routineTimer.reset();
          breakScreenMaxDurationReached = false;
          // update component parameters for each repeat
          advanceScreenPress_2.keys = undefined;
          advanceScreenPress_2.rt = undefined;
          _advanceScreenPress_2_allKeys = [];
          // Run 'Begin Routine' code from endRunTime
          currentLoop.addData("runOffTime", fmriClock.getTime());
          if ((expInfo["startFromRun"] === "2")) {
              console.log("trying to end loop");
              trials.finished = true;
              runs.finished = true;
              continueRoutine = false;
          } else {
              if ((runs.thisN === 1)) {
                  continueRoutine = false;
              }
          }
          
          // Run 'Begin Routine' code from frameOffCode
          frame.setAutoDraw(false);
          
          psychoJS.experiment.addData('breakScreen.started', globalClock.getTime());
          breakScreenMaxDuration = null
          // keep track of which components have finished
          breakScreenComponents = [];
          breakScreenComponents.push(breakText);
          breakScreenComponents.push(advanceScreenPress_2);
          
          for (const thisComponent of breakScreenComponents)
            if ('status' in thisComponent)
              thisComponent.status = PsychoJS.Status.NOT_STARTED;
          return Scheduler.Event.NEXT;
        }
      }
      
      function breakScreenRoutineEachFrame() {
        return async function () {
          //--- Loop for each frame of Routine 'breakScreen' ---
          // get current time
          t = breakScreenClock.getTime();
          frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
          // update/draw components on each frame
          
          // *breakText* updates
          if (t >= 0.0 && breakText.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            breakText.tStart = t;  // (not accounting for frame time here)
            breakText.frameNStart = frameN;  // exact frame index
            
            breakText.setAutoDraw(true);
          }
          
          
          // if breakText is active this frame...
          if (breakText.status === PsychoJS.Status.STARTED) {
          }
          
          
          // *advanceScreenPress_2* updates
          if (t >= 0.0 && advanceScreenPress_2.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            advanceScreenPress_2.tStart = t;  // (not accounting for frame time here)
            advanceScreenPress_2.frameNStart = frameN;  // exact frame index
            
            // keyboard checking is just starting
            psychoJS.window.callOnFlip(function() { advanceScreenPress_2.clock.reset(); });  // t=0 on next screen flip
            psychoJS.window.callOnFlip(function() { advanceScreenPress_2.start(); }); // start on screen flip
            psychoJS.window.callOnFlip(function() { advanceScreenPress_2.clearEvents(); });
          }
          
          // if advanceScreenPress_2 is active this frame...
          if (advanceScreenPress_2.status === PsychoJS.Status.STARTED) {
            let theseKeys = advanceScreenPress_2.getKeys({keyList: ['y','n','left','right','space'], waitRelease: false});
            _advanceScreenPress_2_allKeys = _advanceScreenPress_2_allKeys.concat(theseKeys);
            if (_advanceScreenPress_2_allKeys.length > 0) {
              advanceScreenPress_2.keys = _advanceScreenPress_2_allKeys[_advanceScreenPress_2_allKeys.length - 1].name;  // just the last key pressed
              advanceScreenPress_2.rt = _advanceScreenPress_2_allKeys[_advanceScreenPress_2_allKeys.length - 1].rt;
              advanceScreenPress_2.duration = _advanceScreenPress_2_allKeys[_advanceScreenPress_2_allKeys.length - 1].duration;
              // a response ends the routine
              continueRoutine = false;
            }
          }
          
          // check for quit (typically the Esc key)
          if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
          }
          
          // check if the Routine should terminate
          if (!continueRoutine) {  // a component has requested a forced-end of Routine
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
          }
          
          continueRoutine = false;  // reverts to True if at least one component still running
          for (const thisComponent of breakScreenComponents)
            if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
              continueRoutine = true;
              break;
            }
          
          // refresh the screen if continuing
          if (continueRoutine) {
            return Scheduler.Event.FLIP_REPEAT;
          } else {
            return Scheduler.Event.NEXT;
          }
        };
      }
      
      function breakScreenRoutineEnd(snapshot) {
        return async function () {
          //--- Ending Routine 'breakScreen' ---
          for (const thisComponent of breakScreenComponents) {
            if (typeof thisComponent.setAutoDraw === 'function') {
              thisComponent.setAutoDraw(false);
            }
          }
          psychoJS.experiment.addData('breakScreen.stopped', globalClock.getTime());
          // update the trial handler
          if (currentLoop instanceof MultiStairHandler) {
            currentLoop.addResponse(advanceScreenPress_2.corr, level);
          }
          psychoJS.experiment.addData('advanceScreenPress_2.keys', advanceScreenPress_2.keys);
          if (typeof advanceScreenPress_2.keys !== 'undefined') {  // we had a response
              psychoJS.experiment.addData('advanceScreenPress_2.rt', advanceScreenPress_2.rt);
              psychoJS.experiment.addData('advanceScreenPress_2.duration', advanceScreenPress_2.duration);
              routineTimer.reset();
              }
          
          advanceScreenPress_2.stop();
          // the Routine "breakScreen" was not non-slip safe, so reset the non-slip timer
          routineTimer.reset();
          
          // Routines running outside a loop should always advance the datafile row
          if (currentLoop === psychoJS.experiment) {
            psychoJS.experiment.nextEntry(snapshot);
          }
          return Scheduler.Event.NEXT;
        }
      }
      
      function check6RoutineBegin(snapshot) {
        return async function () {
          TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
          
          //--- Prepare to start Routine 'check6' ---
          t = 0;
          frameN = -1;
          continueRoutine = true; // until we're told otherwise
          // keep track of whether this Routine was forcibly ended
          routineForceEnded = false;
          check6Clock.reset();
          routineTimer.reset();
          check6MaxDurationReached = false;
          // update component parameters for each repeat
          check6ImageLeft.setImage(hg1Img);
          check6ImageRight.setImage(hg2Img);
          // Run 'Begin Routine' code from check6Select
          selectionIndicatorPrac.status = NOT_STARTED;
          t_started = false;
          
          check6Press.keys = undefined;
          check6Press.rt = undefined;
          _check6Press_allKeys = [];
          psychoJS.experiment.addData('check6.started', globalClock.getTime());
          check6MaxDuration = null
          // keep track of which components have finished
          check6Components = [];
          check6Components.push(check6Text);
          check6Components.push(check6ImageLeft);
          check6Components.push(check6ImageRight);
          check6Components.push(check6Press);
          
          for (const thisComponent of check6Components)
            if ('status' in thisComponent)
              thisComponent.status = PsychoJS.Status.NOT_STARTED;
          return Scheduler.Event.NEXT;
        }
      }
      
      function check6RoutineEachFrame() {
        return async function () {
          //--- Loop for each frame of Routine 'check6' ---
          // get current time
          t = check6Clock.getTime();
          frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
          // update/draw components on each frame
          
          // *check6Text* updates
          if (t >= 0.0 && check6Text.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check6Text.tStart = t;  // (not accounting for frame time here)
            check6Text.frameNStart = frameN;  // exact frame index
            
            check6Text.setAutoDraw(true);
          }
          
          
          // if check6Text is active this frame...
          if (check6Text.status === PsychoJS.Status.STARTED) {
          }
          
          
          // *check6ImageLeft* updates
          if (t >= 0.0 && check6ImageLeft.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check6ImageLeft.tStart = t;  // (not accounting for frame time here)
            check6ImageLeft.frameNStart = frameN;  // exact frame index
            
            check6ImageLeft.setAutoDraw(true);
          }
          
          
          // if check6ImageLeft is active this frame...
          if (check6ImageLeft.status === PsychoJS.Status.STARTED) {
          }
          
          
          // *check6ImageRight* updates
          if (t >= 0.0 && check6ImageRight.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check6ImageRight.tStart = t;  // (not accounting for frame time here)
            check6ImageRight.frameNStart = frameN;  // exact frame index
            
            check6ImageRight.setAutoDraw(true);
          }
          
          
          // if check6ImageRight is active this frame...
          if (check6ImageRight.status === PsychoJS.Status.STARTED) {
          }
          
          // Run 'Each Frame' code from check6Select
          if (t_started) {
              if ((t > (t_started + 5))) {
                  continueRoutine = false;
              }
          }
          if ((check6Press.keys && (selectionIndicatorPrac.status === NOT_STARTED))) {
              selectionIndicatorPrac.status = PsychoJS.Status.STARTED;
              selectionIndicatorPrac.setPos(selectionPosition(check6Press.keys));
              selectionIndicatorPrac.setAutoDraw(true);
              t_started = new util.Clock().getTime();
          }
          
          
          // *check6Press* updates
          if (t >= 0.0 && check6Press.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check6Press.tStart = t;  // (not accounting for frame time here)
            check6Press.frameNStart = frameN;  // exact frame index
            
            // keyboard checking is just starting
            psychoJS.window.callOnFlip(function() { check6Press.clock.reset(); });  // t=0 on next screen flip
            psychoJS.window.callOnFlip(function() { check6Press.start(); }); // start on screen flip
            psychoJS.window.callOnFlip(function() { check6Press.clearEvents(); });
          }
          
          // if check6Press is active this frame...
          if (check6Press.status === PsychoJS.Status.STARTED) {
            let theseKeys = check6Press.getKeys({keyList: allowedKeys, waitRelease: false});
            _check6Press_allKeys = _check6Press_allKeys.concat(theseKeys);
            if (_check6Press_allKeys.length > 0) {
              check6Press.keys = _check6Press_allKeys[_check6Press_allKeys.length - 1].name;  // just the last key pressed
              check6Press.rt = _check6Press_allKeys[_check6Press_allKeys.length - 1].rt;
              check6Press.duration = _check6Press_allKeys[_check6Press_allKeys.length - 1].duration;
            }
          }
          
          // check for quit (typically the Esc key)
          if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
          }
          
          // check if the Routine should terminate
          if (!continueRoutine) {  // a component has requested a forced-end of Routine
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
          }
          
          continueRoutine = false;  // reverts to True if at least one component still running
          for (const thisComponent of check6Components)
            if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
              continueRoutine = true;
              break;
            }
          
          // refresh the screen if continuing
          if (continueRoutine) {
            return Scheduler.Event.FLIP_REPEAT;
          } else {
            return Scheduler.Event.NEXT;
          }
        };
      }
      
      function check6RoutineEnd(snapshot) {
        return async function () {
          //--- Ending Routine 'check6' ---
          for (const thisComponent of check6Components) {
            if (typeof thisComponent.setAutoDraw === 'function') {
              thisComponent.setAutoDraw(false);
            }
          }
          psychoJS.experiment.addData('check6.stopped', globalClock.getTime());
          // Run 'End Routine' code from check6Select
          selectionIndicatorPrac.setAutoDraw(false);
          if ((check6Press.keys === "left")) {
              currentLoop.addData("check6", "correct");
          } else {
              currentLoop.addData("check6", "incorrect");
          }
          
          // update the trial handler
          if (currentLoop instanceof MultiStairHandler) {
            currentLoop.addResponse(check6Press.corr, level);
          }
          psychoJS.experiment.addData('check6Press.keys', check6Press.keys);
          if (typeof check6Press.keys !== 'undefined') {  // we had a response
              psychoJS.experiment.addData('check6Press.rt', check6Press.rt);
              psychoJS.experiment.addData('check6Press.duration', check6Press.duration);
              }
          
          check6Press.stop();
          // the Routine "check6" was not non-slip safe, so reset the non-slip timer
          routineTimer.reset();
          
          // Routines running outside a loop should always advance the datafile row
          if (currentLoop === psychoJS.experiment) {
            psychoJS.experiment.nextEntry(snapshot);
          }
          return Scheduler.Event.NEXT;
        }
      }
      
      function check7RoutineBegin(snapshot) {
        return async function () {
          TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
          
          //--- Prepare to start Routine 'check7' ---
          t = 0;
          frameN = -1;
          continueRoutine = true; // until we're told otherwise
          // keep track of whether this Routine was forcibly ended
          routineForceEnded = false;
          check7Clock.reset();
          routineTimer.reset();
          check7MaxDurationReached = false;
          // update component parameters for each repeat
          check7ImageLeft.setImage(lg1Img);
          check7ImageRight.setImage(lg2Img);
          // Run 'Begin Routine' code from check7Select
          selectionIndicatorPrac.status = NOT_STARTED;
          t_started = false;
          
          check7Press.keys = undefined;
          check7Press.rt = undefined;
          _check7Press_allKeys = [];
          psychoJS.experiment.addData('check7.started', globalClock.getTime());
          check7MaxDuration = null
          // keep track of which components have finished
          check7Components = [];
          check7Components.push(check7Text);
          check7Components.push(check7ImageLeft);
          check7Components.push(check7ImageRight);
          check7Components.push(check7Press);
          
          for (const thisComponent of check7Components)
            if ('status' in thisComponent)
              thisComponent.status = PsychoJS.Status.NOT_STARTED;
          return Scheduler.Event.NEXT;
        }
      }
      
      function check7RoutineEachFrame() {
        return async function () {
          //--- Loop for each frame of Routine 'check7' ---
          // get current time
          t = check7Clock.getTime();
          frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
          // update/draw components on each frame
          
          // *check7Text* updates
          if (t >= 0.0 && check7Text.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check7Text.tStart = t;  // (not accounting for frame time here)
            check7Text.frameNStart = frameN;  // exact frame index
            
            check7Text.setAutoDraw(true);
          }
          
          
          // if check7Text is active this frame...
          if (check7Text.status === PsychoJS.Status.STARTED) {
          }
          
          
          // *check7ImageLeft* updates
          if (t >= 0.0 && check7ImageLeft.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check7ImageLeft.tStart = t;  // (not accounting for frame time here)
            check7ImageLeft.frameNStart = frameN;  // exact frame index
            
            check7ImageLeft.setAutoDraw(true);
          }
          
          
          // if check7ImageLeft is active this frame...
          if (check7ImageLeft.status === PsychoJS.Status.STARTED) {
          }
          
          
          // *check7ImageRight* updates
          if (t >= 0.0 && check7ImageRight.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check7ImageRight.tStart = t;  // (not accounting for frame time here)
            check7ImageRight.frameNStart = frameN;  // exact frame index
            
            check7ImageRight.setAutoDraw(true);
          }
          
          
          // if check7ImageRight is active this frame...
          if (check7ImageRight.status === PsychoJS.Status.STARTED) {
          }
          
          // Run 'Each Frame' code from check7Select
          if (t_started) {
              if ((t > (t_started + 5))) {
                  continueRoutine = false;
              }
          }
          if ((check7Press.keys && (selectionIndicatorPrac.status === NOT_STARTED))) {
              selectionIndicatorPrac.status = PsychoJS.Status.STARTED;
              selectionIndicatorPrac.setPos(selectionPosition(check7Press.keys));
              selectionIndicatorPrac.setAutoDraw(true);
              t_started = new util.Clock().getTime();
          }
          
          
          // *check7Press* updates
          if (t >= 0.0 && check7Press.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check7Press.tStart = t;  // (not accounting for frame time here)
            check7Press.frameNStart = frameN;  // exact frame index
            
            // keyboard checking is just starting
            psychoJS.window.callOnFlip(function() { check7Press.clock.reset(); });  // t=0 on next screen flip
            psychoJS.window.callOnFlip(function() { check7Press.start(); }); // start on screen flip
            psychoJS.window.callOnFlip(function() { check7Press.clearEvents(); });
          }
          
          // if check7Press is active this frame...
          if (check7Press.status === PsychoJS.Status.STARTED) {
            let theseKeys = check7Press.getKeys({keyList: allowedKeys, waitRelease: false});
            _check7Press_allKeys = _check7Press_allKeys.concat(theseKeys);
            if (_check7Press_allKeys.length > 0) {
              check7Press.keys = _check7Press_allKeys[_check7Press_allKeys.length - 1].name;  // just the last key pressed
              check7Press.rt = _check7Press_allKeys[_check7Press_allKeys.length - 1].rt;
              check7Press.duration = _check7Press_allKeys[_check7Press_allKeys.length - 1].duration;
            }
          }
          
          // check for quit (typically the Esc key)
          if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
          }
          
          // check if the Routine should terminate
          if (!continueRoutine) {  // a component has requested a forced-end of Routine
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
          }
          
          continueRoutine = false;  // reverts to True if at least one component still running
          for (const thisComponent of check7Components)
            if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
              continueRoutine = true;
              break;
            }
          
          // refresh the screen if continuing
          if (continueRoutine) {
            return Scheduler.Event.FLIP_REPEAT;
          } else {
            return Scheduler.Event.NEXT;
          }
        };
      }
      
      function check7RoutineEnd(snapshot) {
        return async function () {
          //--- Ending Routine 'check7' ---
          for (const thisComponent of check7Components) {
            if (typeof thisComponent.setAutoDraw === 'function') {
              thisComponent.setAutoDraw(false);
            }
          }
          psychoJS.experiment.addData('check7.stopped', globalClock.getTime());
          // Run 'End Routine' code from check7Select
          selectionIndicatorPrac.setAutoDraw(false);
          if ((check7Press.keys === "right")) {
              currentLoop.addData("check7", "correct");
          } else {
              currentLoop.addData("check7", "incorrect");
          }
          
          // update the trial handler
          if (currentLoop instanceof MultiStairHandler) {
            currentLoop.addResponse(check7Press.corr, level);
          }
          psychoJS.experiment.addData('check7Press.keys', check7Press.keys);
          if (typeof check7Press.keys !== 'undefined') {  // we had a response
              psychoJS.experiment.addData('check7Press.rt', check7Press.rt);
              psychoJS.experiment.addData('check7Press.duration', check7Press.duration);
              }
          
          check7Press.stop();
          // the Routine "check7" was not non-slip safe, so reset the non-slip timer
          routineTimer.reset();
          
          // Routines running outside a loop should always advance the datafile row
          if (currentLoop === psychoJS.experiment) {
            psychoJS.experiment.nextEntry(snapshot);
          }
          return Scheduler.Event.NEXT;
        }
      }
      
      function check8RoutineBegin(snapshot) {
        return async function () {
          TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
          
          //--- Prepare to start Routine 'check8' ---
          t = 0;
          frameN = -1;
          continueRoutine = true; // until we're told otherwise
          // keep track of whether this Routine was forcibly ended
          routineForceEnded = false;
          check8Clock.reset();
          routineTimer.reset();
          check8MaxDurationReached = false;
          // update component parameters for each repeat
          check8ImageLeft.setImage(hl1Img);
          check8ImageRight.setImage(hl2Img);
          // Run 'Begin Routine' code from check8Select
          selectionIndicatorPrac.status = NOT_STARTED;
          t_started = false;
          
          check8Press.keys = undefined;
          check8Press.rt = undefined;
          _check8Press_allKeys = [];
          psychoJS.experiment.addData('check8.started', globalClock.getTime());
          check8MaxDuration = null
          // keep track of which components have finished
          check8Components = [];
          check8Components.push(check8Text);
          check8Components.push(check8ImageLeft);
          check8Components.push(check8ImageRight);
          check8Components.push(check8Press);
          
          for (const thisComponent of check8Components)
            if ('status' in thisComponent)
              thisComponent.status = PsychoJS.Status.NOT_STARTED;
          return Scheduler.Event.NEXT;
        }
      }
      
      function check8RoutineEachFrame() {
        return async function () {
          //--- Loop for each frame of Routine 'check8' ---
          // get current time
          t = check8Clock.getTime();
          frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
          // update/draw components on each frame
          
          // *check8Text* updates
          if (t >= 0.0 && check8Text.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check8Text.tStart = t;  // (not accounting for frame time here)
            check8Text.frameNStart = frameN;  // exact frame index
            
            check8Text.setAutoDraw(true);
          }
          
          
          // if check8Text is active this frame...
          if (check8Text.status === PsychoJS.Status.STARTED) {
          }
          
          
          // *check8ImageLeft* updates
          if (t >= 0.0 && check8ImageLeft.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check8ImageLeft.tStart = t;  // (not accounting for frame time here)
            check8ImageLeft.frameNStart = frameN;  // exact frame index
            
            check8ImageLeft.setAutoDraw(true);
          }
          
          
          // if check8ImageLeft is active this frame...
          if (check8ImageLeft.status === PsychoJS.Status.STARTED) {
          }
          
          
          // *check8ImageRight* updates
          if (t >= 0.0 && check8ImageRight.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check8ImageRight.tStart = t;  // (not accounting for frame time here)
            check8ImageRight.frameNStart = frameN;  // exact frame index
            
            check8ImageRight.setAutoDraw(true);
          }
          
          
          // if check8ImageRight is active this frame...
          if (check8ImageRight.status === PsychoJS.Status.STARTED) {
          }
          
          // Run 'Each Frame' code from check8Select
          if (t_started) {
              if ((t > (t_started + 5))) {
                  continueRoutine = false;
              }
          }
          if ((check8Press.keys && (selectionIndicatorPrac.status === NOT_STARTED))) {
              selectionIndicatorPrac.status = PsychoJS.Status.STARTED;
              selectionIndicatorPrac.setPos(selectionPosition(check8Press.keys));
              selectionIndicatorPrac.setAutoDraw(true);
              t_started = new util.Clock().getTime();
          }
          
          
          // *check8Press* updates
          if (t >= 0.0 && check8Press.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check8Press.tStart = t;  // (not accounting for frame time here)
            check8Press.frameNStart = frameN;  // exact frame index
            
            // keyboard checking is just starting
            psychoJS.window.callOnFlip(function() { check8Press.clock.reset(); });  // t=0 on next screen flip
            psychoJS.window.callOnFlip(function() { check8Press.start(); }); // start on screen flip
            psychoJS.window.callOnFlip(function() { check8Press.clearEvents(); });
          }
          
          // if check8Press is active this frame...
          if (check8Press.status === PsychoJS.Status.STARTED) {
            let theseKeys = check8Press.getKeys({keyList: allowedKeys, waitRelease: false});
            _check8Press_allKeys = _check8Press_allKeys.concat(theseKeys);
            if (_check8Press_allKeys.length > 0) {
              check8Press.keys = _check8Press_allKeys[_check8Press_allKeys.length - 1].name;  // just the last key pressed
              check8Press.rt = _check8Press_allKeys[_check8Press_allKeys.length - 1].rt;
              check8Press.duration = _check8Press_allKeys[_check8Press_allKeys.length - 1].duration;
            }
          }
          
          // check for quit (typically the Esc key)
          if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
          }
          
          // check if the Routine should terminate
          if (!continueRoutine) {  // a component has requested a forced-end of Routine
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
          }
          
          continueRoutine = false;  // reverts to True if at least one component still running
          for (const thisComponent of check8Components)
            if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
              continueRoutine = true;
              break;
            }
          
          // refresh the screen if continuing
          if (continueRoutine) {
            return Scheduler.Event.FLIP_REPEAT;
          } else {
            return Scheduler.Event.NEXT;
          }
        };
      }
      
      function check8RoutineEnd(snapshot) {
        return async function () {
          //--- Ending Routine 'check8' ---
          for (const thisComponent of check8Components) {
            if (typeof thisComponent.setAutoDraw === 'function') {
              thisComponent.setAutoDraw(false);
            }
          }
          psychoJS.experiment.addData('check8.stopped', globalClock.getTime());
          // Run 'End Routine' code from check8Select
          selectionIndicatorPrac.setAutoDraw(false);
          if ((check8Press.keys === "right")) {
              currentLoop.addData("check8", "correct");
          } else {
              currentLoop.addData("check8", "incorrect");
          }
          
          // update the trial handler
          if (currentLoop instanceof MultiStairHandler) {
            currentLoop.addResponse(check8Press.corr, level);
          }
          psychoJS.experiment.addData('check8Press.keys', check8Press.keys);
          if (typeof check8Press.keys !== 'undefined') {  // we had a response
              psychoJS.experiment.addData('check8Press.rt', check8Press.rt);
              psychoJS.experiment.addData('check8Press.duration', check8Press.duration);
              }
          
          check8Press.stop();
          // the Routine "check8" was not non-slip safe, so reset the non-slip timer
          routineTimer.reset();
          
          // Routines running outside a loop should always advance the datafile row
          if (currentLoop === psychoJS.experiment) {
            psychoJS.experiment.nextEntry(snapshot);
          }
          return Scheduler.Event.NEXT;
        }
      }
      
      function check9RoutineBegin(snapshot) {
        return async function () {
          TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
          
          //--- Prepare to start Routine 'check9' ---
          t = 0;
          frameN = -1;
          continueRoutine = true; // until we're told otherwise
          // keep track of whether this Routine was forcibly ended
          routineForceEnded = false;
          check9Clock.reset();
          routineTimer.reset();
          check9MaxDurationReached = false;
          // update component parameters for each repeat
          check9ImageLeft.setImage(ll1Img);
          check9ImageRight.setImage(ll2Img);
          // Run 'Begin Routine' code from check9Select
          selectionIndicatorPrac.status = NOT_STARTED;
          t_started = false;
          
          check9Press.keys = undefined;
          check9Press.rt = undefined;
          _check9Press_allKeys = [];
          psychoJS.experiment.addData('check9.started', globalClock.getTime());
          check9MaxDuration = null
          // keep track of which components have finished
          check9Components = [];
          check9Components.push(check9Text);
          check9Components.push(check9ImageLeft);
          check9Components.push(check9ImageRight);
          check9Components.push(check9Press);
          
          for (const thisComponent of check9Components)
            if ('status' in thisComponent)
              thisComponent.status = PsychoJS.Status.NOT_STARTED;
          return Scheduler.Event.NEXT;
        }
      }
      
      function check9RoutineEachFrame() {
        return async function () {
          //--- Loop for each frame of Routine 'check9' ---
          // get current time
          t = check9Clock.getTime();
          frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
          // update/draw components on each frame
          
          // *check9Text* updates
          if (t >= 0.0 && check9Text.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check9Text.tStart = t;  // (not accounting for frame time here)
            check9Text.frameNStart = frameN;  // exact frame index
            
            check9Text.setAutoDraw(true);
          }
          
          
          // if check9Text is active this frame...
          if (check9Text.status === PsychoJS.Status.STARTED) {
          }
          
          
          // *check9ImageLeft* updates
          if (t >= 0.0 && check9ImageLeft.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check9ImageLeft.tStart = t;  // (not accounting for frame time here)
            check9ImageLeft.frameNStart = frameN;  // exact frame index
            
            check9ImageLeft.setAutoDraw(true);
          }
          
          
          // if check9ImageLeft is active this frame...
          if (check9ImageLeft.status === PsychoJS.Status.STARTED) {
          }
          
          
          // *check9ImageRight* updates
          if (t >= 0.0 && check9ImageRight.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check9ImageRight.tStart = t;  // (not accounting for frame time here)
            check9ImageRight.frameNStart = frameN;  // exact frame index
            
            check9ImageRight.setAutoDraw(true);
          }
          
          
          // if check9ImageRight is active this frame...
          if (check9ImageRight.status === PsychoJS.Status.STARTED) {
          }
          
          // Run 'Each Frame' code from check9Select
          if (t_started) {
              if ((t > (t_started + 5))) {
                  continueRoutine = false;
              }
          }
          if ((check9Press.keys && (selectionIndicatorPrac.status === NOT_STARTED))) {
              selectionIndicatorPrac.status = PsychoJS.Status.STARTED;
              selectionIndicatorPrac.setPos(selectionPosition(check9Press.keys));
              selectionIndicatorPrac.setAutoDraw(true);
              t_started = new util.Clock().getTime();
          }
          
          
          // *check9Press* updates
          if (t >= 0.0 && check9Press.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            check9Press.tStart = t;  // (not accounting for frame time here)
            check9Press.frameNStart = frameN;  // exact frame index
            
            // keyboard checking is just starting
            psychoJS.window.callOnFlip(function() { check9Press.clock.reset(); });  // t=0 on next screen flip
            psychoJS.window.callOnFlip(function() { check9Press.start(); }); // start on screen flip
            psychoJS.window.callOnFlip(function() { check9Press.clearEvents(); });
          }
          
          // if check9Press is active this frame...
          if (check9Press.status === PsychoJS.Status.STARTED) {
            let theseKeys = check9Press.getKeys({keyList: allowedKeys, waitRelease: false});
            _check9Press_allKeys = _check9Press_allKeys.concat(theseKeys);
            if (_check9Press_allKeys.length > 0) {
              check9Press.keys = _check9Press_allKeys[_check9Press_allKeys.length - 1].name;  // just the last key pressed
              check9Press.rt = _check9Press_allKeys[_check9Press_allKeys.length - 1].rt;
              check9Press.duration = _check9Press_allKeys[_check9Press_allKeys.length - 1].duration;
            }
          }
          
          // check for quit (typically the Esc key)
          if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
          }
          
          // check if the Routine should terminate
          if (!continueRoutine) {  // a component has requested a forced-end of Routine
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
          }
          
          continueRoutine = false;  // reverts to True if at least one component still running
          for (const thisComponent of check9Components)
            if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
              continueRoutine = true;
              break;
            }
          
          // refresh the screen if continuing
          if (continueRoutine) {
            return Scheduler.Event.FLIP_REPEAT;
          } else {
            return Scheduler.Event.NEXT;
          }
        };
      }
      
      function check9RoutineEnd(snapshot) {
        return async function () {
          //--- Ending Routine 'check9' ---
          for (const thisComponent of check9Components) {
            if (typeof thisComponent.setAutoDraw === 'function') {
              thisComponent.setAutoDraw(false);
            }
          }
          psychoJS.experiment.addData('check9.stopped', globalClock.getTime());
          // Run 'End Routine' code from check9Select
          selectionIndicatorPrac.setAutoDraw(false);
          if ((check9Press.keys === "left")) {
              currentLoop.addData("check9", "correct");
          } else {
              currentLoop.addData("check9", "incorrect");
          }
          
          // update the trial handler
          if (currentLoop instanceof MultiStairHandler) {
            currentLoop.addResponse(check9Press.corr, level);
          }
          psychoJS.experiment.addData('check9Press.keys', check9Press.keys);
          if (typeof check9Press.keys !== 'undefined') {  // we had a response
              psychoJS.experiment.addData('check9Press.rt', check9Press.rt);
              psychoJS.experiment.addData('check9Press.duration', check9Press.duration);
              }
          
          check9Press.stop();
          // the Routine "check9" was not non-slip safe, so reset the non-slip timer
          routineTimer.reset();
          
          // Routines running outside a loop should always advance the datafile row
          if (currentLoop === psychoJS.experiment) {
            psychoJS.experiment.nextEntry(snapshot);
          }
          return Scheduler.Event.NEXT;
        }
      }
      
      function doneRoutineBegin(snapshot) {
        return async function () {
          TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
          
          //--- Prepare to start Routine 'done' ---
          t = 0;
          frameN = -1;
          continueRoutine = true; // until we're told otherwise
          // keep track of whether this Routine was forcibly ended
          routineForceEnded = false;
          doneClock.reset();
          routineTimer.reset();
          doneMaxDurationReached = false;
          // update component parameters for each repeat
          endTaskPress.keys = undefined;
          endTaskPress.rt = undefined;
          _endTaskPress_allKeys = [];
          psychoJS.experiment.addData('done.started', globalClock.getTime());
          doneMaxDuration = null
          // keep track of which components have finished
          doneComponents = [];
          doneComponents.push(endExperiment);
          doneComponents.push(endTaskPress);
          
          for (const thisComponent of doneComponents)
            if ('status' in thisComponent)
              thisComponent.status = PsychoJS.Status.NOT_STARTED;
          return Scheduler.Event.NEXT;
        }
      }
      
      function doneRoutineEachFrame() {
        return async function () {
          //--- Loop for each frame of Routine 'done' ---
          // get current time
          t = doneClock.getTime();
          frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
          // update/draw components on each frame
          
          // *endExperiment* updates
          if (t >= 0.0 && endExperiment.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            endExperiment.tStart = t;  // (not accounting for frame time here)
            endExperiment.frameNStart = frameN;  // exact frame index
            
            endExperiment.setAutoDraw(true);
          }
          
          
          // if endExperiment is active this frame...
          if (endExperiment.status === PsychoJS.Status.STARTED) {
          }
          
          
          // *endTaskPress* updates
          if (t >= 0.0 && endTaskPress.status === PsychoJS.Status.NOT_STARTED) {
            // keep track of start time/frame for later
            endTaskPress.tStart = t;  // (not accounting for frame time here)
            endTaskPress.frameNStart = frameN;  // exact frame index
            
            // keyboard checking is just starting
            psychoJS.window.callOnFlip(function() { endTaskPress.clock.reset(); });  // t=0 on next screen flip
            psychoJS.window.callOnFlip(function() { endTaskPress.start(); }); // start on screen flip
            psychoJS.window.callOnFlip(function() { endTaskPress.clearEvents(); });
          }
          
          // if endTaskPress is active this frame...
          if (endTaskPress.status === PsychoJS.Status.STARTED) {
            let theseKeys = endTaskPress.getKeys({keyList: ['y','n','left','right','space'], waitRelease: false});
            _endTaskPress_allKeys = _endTaskPress_allKeys.concat(theseKeys);
            if (_endTaskPress_allKeys.length > 0) {
              endTaskPress.keys = _endTaskPress_allKeys[_endTaskPress_allKeys.length - 1].name;  // just the last key pressed
              endTaskPress.rt = _endTaskPress_allKeys[_endTaskPress_allKeys.length - 1].rt;
              endTaskPress.duration = _endTaskPress_allKeys[_endTaskPress_allKeys.length - 1].duration;
              // a response ends the routine
              continueRoutine = false;
            }
          }
          
          // check for quit (typically the Esc key)
          if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
            return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
          }
          
          // check if the Routine should terminate
          if (!continueRoutine) {  // a component has requested a forced-end of Routine
            routineForceEnded = true;
            return Scheduler.Event.NEXT;
          }
          
          continueRoutine = false;  // reverts to True if at least one component still running
          for (const thisComponent of doneComponents)
            if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
              continueRoutine = true;
              break;
            }
          
          // refresh the screen if continuing
          if (continueRoutine) {
            return Scheduler.Event.FLIP_REPEAT;
          } else {
            return Scheduler.Event.NEXT;
          }
        };
      }
      
      function doneRoutineEnd(snapshot) {
        return async function () {
          //--- Ending Routine 'done' ---
          for (const thisComponent of doneComponents) {
            if (typeof thisComponent.setAutoDraw === 'function') {
              thisComponent.setAutoDraw(false);
            }
          }
          psychoJS.experiment.addData('done.stopped', globalClock.getTime());
          // update the trial handler
          if (currentLoop instanceof MultiStairHandler) {
            currentLoop.addResponse(endTaskPress.corr, level);
          }
          psychoJS.experiment.addData('endTaskPress.keys', endTaskPress.keys);
          if (typeof endTaskPress.keys !== 'undefined') {  // we had a response
              psychoJS.experiment.addData('endTaskPress.rt', endTaskPress.rt);
              psychoJS.experiment.addData('endTaskPress.duration', endTaskPress.duration);
              routineTimer.reset();
              }
          
          endTaskPress.stop();
          // the Routine "done" was not non-slip safe, so reset the non-slip timer
          routineTimer.reset();
          
          // Routines running outside a loop should always advance the datafile row
          if (currentLoop === psychoJS.experiment) {
            psychoJS.experiment.nextEntry(snapshot);
          }
          return Scheduler.Event.NEXT;
        }
      }
      
      function importConditions(currentLoop) {
        return async function () {
          psychoJS.importAttributes(currentLoop.getCurrentTrial());
          return Scheduler.Event.NEXT;
          };
      }
      
      async function quitPsychoJS(message, isCompleted) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry();
        }
        psychoJS.window.close();
        psychoJS.quit({message: message, isCompleted: isCompleted});
        
        return Scheduler.Event.QUIT;
      }
