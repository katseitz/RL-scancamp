library(shiny)
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)
library(DT)
library(zoo)
library(stringr)

# UI
ui <- fluidPage(
  titlePanel("Scan Camp Reward Learning Multi-Participant Trial Analysis Dashboard"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload PsychoPy CSV File",
                accept = c(".csv")),
      uiOutput("participant_ui"),
      uiOutput("trial_ui"),
      hr(),
      helpText("Instructions:"),
      helpText("1. Upload your PsychoPy CSV output file."),
      helpText("2. Select a participant to load their trials."),
      helpText("3. Select a trial to view detailed plots."),
      helpText("4. Navigate the tabs for visualizations and data tables."),
      hr(),
      helpText("Variables of Interest:"),
      tags$ul(
        tags$li("cue_onset, cue_offset, cue_duration: Timing of cue presentation."),
        tags$li("outcome_onset, outcome_offset, outcome_duration: Timing of outcome display."),
        tags$li("fixation_onset: Start of fixation period."),
        tags$li("optedFor: Participant's response (left/right)."),
        tags$li("optimalResponse: Expected correct response."),
        tags$li("Accuracy: Whether the response was correct."),
        tags$li("OutcomeValue: The outcome (reward/feedback).")
      )
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Timeline Plot", plotOutput("timelinePlot", height = "400px")),
        tabPanel("Trial Table", DTOutput("trialTable")),
        tabPanel("Accuracy Plot", plotOutput("accuracyPlot", height = "400px")),
        tabPanel("Rolling Avg Plot", plotOutput("rollingAvgPlot", height = "400px")),
        tabPanel("Outcome Plot", plotOutput("outcomePlot", height = "400px")),
        tabPanel("Image Selection Histogram", plotOutput("selectionHistogram", height = "400px")),
        tabPanel("Cumulative Bank", plotOutput("bankPlot", height = "400px")),
        tabPanel("Feedback by Trial", plotOutput("rewardPlot", height = "400px")),
        tabPanel("Optimal Choice Rate", plotOutput("optimalityPlot", height = "400px")),
        tabPanel("Reaction Time", plotOutput("rtPlot", height = "400px"))
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  data <- reactive({
    req(input$file)
    df <- read_csv(input$file$datapath)
    
    df <- df %>%
      mutate(
        Participant = as.character(participant),
        cue_onset = round(as.numeric(cueOnTime), 2),
        cue_offset = round(as.numeric(cueOffTime), 2),
        cue_duration = cue_offset - cue_onset,
        outcome_onset = round(as.numeric(feedbackTime), 2),
        outcome_offset = round(as.numeric(feedbackOffTime), 2),
        outcome_duration = outcome_offset - outcome_onset,
        fixation_onset = round(as.numeric(fixationTime), 2),
        trial_id = as.numeric(trialOrder),
        OutcomeValue = outcome,
        ImageID = case_when(
          grepl("img1", leftImage) | grepl("img1", rightImage) ~ "img1",
          grepl("img2", leftImage) | grepl("img2", rightImage) ~ "img2",
          grepl("img3", leftImage) | grepl("img3", rightImage) ~ "img3",
          grepl("img4", leftImage) | grepl("img4", rightImage) ~ "img4",
          grepl("img5", leftImage) | grepl("img5", rightImage) ~ "img5",
          grepl("img6", leftImage) | grepl("img6", rightImage) ~ "img6",
          grepl("img7", leftImage) | grepl("img7", rightImage) ~ "img7",
          grepl("img8", leftImage) | grepl("img8", rightImage) ~ "img8",
          TRUE ~ "Other"
        ),
        ImagePair = case_when(
          str_detect(optedImg, "1.png") | str_detect(optedImg, "2.png") ~ "Pair 1",
          str_detect(optedImg, "3.png") | str_detect(optedImg, "4.png") ~ "Pair 2",
          str_detect(optedImg, "5.png") | str_detect(optedImg, "6.png") ~ "Pair 3",
          str_detect(optedImg, "7.png") | str_detect(optedImg, "8.png") ~ "Pair 4",
          TRUE ~ "Other"
        ),
        ChosenImage = ifelse(optedFor == "left", leftImage, rightImage)
      ) %>%
      select(Participant, condition, trial_id, cue_onset, cue_offset, cue_duration,
             outcome_onset, outcome_offset, outcome_duration,
             fixation_onset, optedFor, optimalResponse, accuracy,
             OutcomeValue, leftImage, rightImage, ImageID, optedImg,
             ImagePair) %>%
      filter(!is.na(cue_onset))
    
    return(df)
  })
  
  output$participant_ui <- renderUI({
    req(data())
    selectInput("participant", "Select Participant", choices = unique(data()$Participant))
  })
  
  output$trial_ui <- renderUI({
    req(input$participant)
    df <- data() %>% filter(Participant == input$participant)
    selectInput("trial", "Select Trial", choices = unique(df$trial_id))
  })
  
  participant_data <- reactive({
    req(input$participant)
    data() %>% filter(Participant == input$participant)
  })
  
  output$trialTable <- renderDT({
    datatable(participant_data(), options = list(pageLength = 10, scrollX = TRUE),
              class = "display compact")
  })
  
  output$timelinePlot <- renderPlot({
    req(input$trial)
    trial_data <- participant_data() %>% filter(trial_id == as.numeric(input$trial))
    
    events <- tribble(
      ~event, ~onset, ~duration,
      "Cue", trial_data$cue_onset, trial_data$cue_duration,
      "Outcome", trial_data$outcome_onset, trial_data$outcome_duration,
      "Fixation", trial_data$fixation_onset, 1
    )
    
    ggplot(events, aes(x = onset, xend = onset + duration, y = event, yend = event)) +
      geom_segment(size = 8, color = "#0073C2FF") +
      theme_minimal(base_size = 14) +
      labs(title = paste("Timeline for Trial", trial_data$trial_id, "(Participant:", trial_data$Participant, ")"),
           x = "Time (seconds)", y = "Event")
  })
  
  output$accuracyPlot <- renderPlot({
    df <- participant_data()
    
    ggplot(df, aes(x = trial_id, y = accuracy)) +
      geom_point(color = "#E64B35FF", size = 2) +
      geom_line(color = "#E64B35FF") +
      scale_y_continuous(breaks = c(0, 1), labels = c("Incorrect", "Correct")) +
      labs(title = paste("Accuracy Across Trials (Participant:", input$participant, ")"),
           x = "Trial ID", y = "Accuracy") +
      theme_minimal(base_size = 14)
  })
  
  output$rollingAvgPlot <- renderPlot({
    df <- participant_data() %>%
      arrange(trial_id) %>%
      mutate(rolling_avg = zoo::rollmean(accuracy, k = 5, fill = NA, align = "right"))
    
    ggplot(df, aes(x = trial_id)) +
      geom_line(aes(y = rolling_avg), color = "#00A087FF", size = 1.2) +
      geom_point(aes(y = rolling_avg), color = "#00A087FF", size = 2) +
      ylim(0, 1) +
      labs(title = paste("5-Trial Rolling Average Accuracy (Participant:", input$participant, ")"),
           x = "Trial ID", y = "Rolling Avg Accuracy") +
      theme_minimal(base_size = 14)
  })
  
  output$outcomePlot <- renderPlot({
    df <- participant_data()
    
    ggplot(df, aes(x = trial_id, y = OutcomeValue)) +
      geom_col(fill = "#3C5488FF") +
      labs(title = paste("Outcome per Trial (Participant:", input$participant, ")"),
           x = "Trial ID", y = "Outcome Value") +
      theme_minimal(base_size = 14)
  })
  
  output$selectionHistogram <- renderPlot({
    ggplot(df_clean, aes(x = optedImg, fill = condition)) +
      geom_bar(position = position_dodge(width = 0.9)) +
      labs(title = paste("Selection Frequency by Image (Grouped by Pair) - Participant:"),
           x = "Chosen Image", y = "Count", fill = "Image Pair") +
      theme_minimal(base_size = 14)
  })
  output$bankPlot <- renderPlot({
    df <- participant_data()
    if (!"bank" %in% colnames(df)) return()
    
    ggplot(df, aes(x = trial_id, y = bank)) +
      geom_line(color = "#4DBBD5FF", size = 1.2) +
      labs(title = paste("Cumulative Bank Over Trials (Participant:", input$participant, ")"),
           x = "Trial ID", y = "Bank Value") +
      theme_minimal(base_size = 14)
  })
  
  output$rewardPlot <- renderPlot({
    df <- participant_data()
    if (!"OutcomeValue" %in% colnames(df)) return()
    
    ggplot(df, aes(x = trial_id, y = OutcomeValue, color = condition)) +
      geom_point(alpha = 0.7) +
      geom_smooth(se = FALSE) +
      labs(title = paste("Reward Feedback by Trial (Participant:", input$participant, ")"),
           x = "Trial ID", y = "Reward/Feedback Value") +
      theme_minimal(base_size = 14)
  })
  
  output$optimalityPlot <- renderPlot({
    df <- participant_data()
    if (!"optimalResponse" %in% colnames(df) || !"optedFor" %in% colnames(df)) return()
    
    df <- df %>%
      mutate(optimal = as.numeric(optedFor == optimalResponse)) %>%
      arrange(trial_id) %>%
      mutate(rolling_opt = zoo::rollmean(optimal, k = 5, fill = NA, align = "right"))
    
    ggplot(df, aes(x = trial_id, y = rolling_opt)) +
      geom_line(color = "#F39B7FFF", size = 1.2) +
      labs(title = paste("5-Trial Rolling Optimal Choice Rate (Participant:", input$participant, ")"),
           x = "Trial ID", y = "Proportion Optimal") +
      ylim(0, 1) +
      theme_minimal(base_size = 14)
  })
  
  output$rtPlot <- renderPlot({
    df <- participant_data()
    rt_cols <- grep("rt", names(df), value = TRUE)
    rt_col <- rt_cols[1]
    if (is.null(rt_col)) return()
    
    df <- df %>% rename(rt = all_of(rt_col))
    
    ggplot(df, aes(x = trial_id, y = rt)) +
      geom_point(alpha = 0.6, color = "#8491B4FF") +
      geom_smooth(se = FALSE, color = "#8491B4FF") +
      labs(title = paste("Reaction Time by Trial (Participant:", input$participant, ")"),
           x = "Trial ID", y = "RT (seconds)") +
      theme_minimal(base_size = 14)
  })
  
}

shinyApp(ui, server)
