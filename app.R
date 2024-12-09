library(shiny)
library(shinythemes)
library(randomForest)
library(ggplot2)
library(pdp)

ui <- fluidPage(
  theme = shinytheme("cerulean"), 
  
  titlePanel("Random Forest Model for Final Grade Prediction"),
  
  sidebarLayout(
    sidebarPanel(
      style = "background-color: #E6F7FF;",
      checkboxGroupInput("selected_vars", "Select Variables for Modeling:", 
                         choices = NULL), 
      sliderInput("split_ratio", "Training Set Ratio:", min = 0.5, max = 0.9, value = 0.8, step = 0.05),
      actionButton("train_model", "Train Model"),
      selectInput("pdp_var1", "Select the First Variable for PDP:", choices = NULL),
      selectInput("pdp_var2", "Select the Second Variable for PDP:", choices = NULL)
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Model Overview", 
                 verbatimTextOutput("accuracy"),
                 plotOutput("var_importance_plot")),
        tabPanel("Partial Dependence Plots",
                 plotOutput("pdp_plot1"),
                 plotOutput("pdp_plot2"))
      )
    )
  )
)

server <- function(input, output, session) {
  data <- reactive({
    read.csv("spring23.csv")
  })
  
  observe({
    df <- data()
    features <- colnames(df)[2:which(names(df) == "ReportPeerFeedback")]
    updateCheckboxGroupInput(session, "selected_vars", 
                             choices = features, 
                             selected = features)
  })
  
  observe({
    req(input$selected_vars)
    updateSelectInput(session, "pdp_var1", 
                      choices = input$selected_vars, 
                      selected = input$selected_vars[1])
    updateSelectInput(session, "pdp_var2", 
                      choices = input$selected_vars, 
                      selected = input$selected_vars[2])
  })
  
  # Train the Random Forest model
  model_results <- eventReactive(input$train_model, {
    req(input$selected_vars)
    
    df <- data()
    X <- df[, input$selected_vars, drop = FALSE]
    Y <- as.factor(df$Grade)
    
    set.seed(123)
    trainIndex <- caret::createDataPartition(Y, p = input$split_ratio, list = FALSE)
    X_train <- X[trainIndex, ]
    X_test <- X[-trainIndex, ]
    Y_train <- Y[trainIndex]
    Y_test <- Y[-trainIndex]
    
    rf_model <- randomForest(x = X_train, y = Y_train, importance = TRUE, ntree = 500)
    predictions <- predict(rf_model, newdata = X_test)
    accuracy <- round(sum(predictions == Y_test) / length(Y_test) * 100, 2)
    
    list(rf_model = rf_model, X_train = X_train, accuracy = accuracy)
  })
  
  # Display model accuracy
  output$accuracy <- renderPrint({
    req(model_results())
    acc <- model_results()$accuracy
    paste("The model achieved an accuracy of", acc, "% on the test data.")
  })
  
  # Plot variable importance
  output$var_importance_plot <- renderPlot({
    req(model_results())
    rf_model <- model_results()$rf_model
    importance_df <- as.data.frame(randomForest::importance(rf_model))
    importance_df <- tibble::rownames_to_column(importance_df, "Variable")
    
    ggplot(importance_df, aes(x = reorder(Variable, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
      geom_bar(stat = "identity", fill = "#0073C2FF") +
      coord_flip() +
      labs(title = "Variable Importance", x = "Variables", y = "Mean Decrease in Accuracy") +
      theme_minimal()
  })
  
  # Plot the first PDP
  output$pdp_plot1 <- renderPlot({
    req(model_results())
    req(input$pdp_var1)
    
    rf_model <- model_results()$rf_model
    X_train <- model_results()$X_train
    feature <- input$pdp_var1
    
    pdp_data <- partial(object = rf_model, pred.var = feature, train = X_train, grid.resolution = 50)
    
    ggplot(pdp_data, aes(x = !!sym(feature), y = yhat)) +
      geom_line(color = "lightblue", size = 1.2) +
      geom_point(color = "#FC8D62", size = 2) +
      theme_minimal() +
      labs(
        title = paste("Partial Dependence of", feature),
        x = feature,
        y = "Predicted Value"
      ) +
      theme(
        text = element_text(size = 14),
        plot.title = element_text(hjust = 0.5, face = "bold")
      )
  })
  
  # Plot the second PDP
  output$pdp_plot2 <- renderPlot({
    req(model_results())
    req(input$pdp_var2)
    
    rf_model <- model_results()$rf_model
    X_train <- model_results()$X_train
    feature <- input$pdp_var2

    pdp_data <- partial(object = rf_model, pred.var = feature, train = X_train, grid.resolution = 50)
    
    ggplot(pdp_data, aes(x = !!sym(feature), y = yhat)) +
      geom_line(color = "#66C2A5", size = 1.2) +
      geom_point(color = "#FC8D62", size = 2) +
      theme_minimal() +
      labs(
        title = paste("Partial Dependence of", feature),
        x = feature,
        y = "Predicted Value"
      ) +
      theme(
        text = element_text(size = 14),
        plot.title = element_text(hjust = 0.5, face = "bold")
      )
  })
}

shinyApp(ui = ui, server = server)
