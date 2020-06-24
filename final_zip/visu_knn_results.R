#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(dplyr)
library(tidyr)
library(ggplot2)
library(viridis)

data_bin <- read.csv("output/knn/optim_history_bin4.csv", header=FALSE)%>% select(-V1)
colnames(data_bin) <- c("k", "alpha_animal", "alpha_experiment", "accuracy", "dist_animal","dist_experiment","dist_chem")
data_bin$k <- as.factor(data_bin$k)

data_mult <- read.csv("output/knn/optim_history_mult4.csv", header=FALSE)%>% select(-V1)
colnames(data_mult) <- c("k", "alpha_animal", "alpha_experiment", "accuracy", "dist_animal","dist_experiment","dist_chem")
data_mult$k <- as.factor(data_mult$k)


# Prepare plots
gg1 <- ggplot(data_bin, aes(y=accuracy, color=k, shape=dist_chem)) + 
    geom_point(aes(x=alpha_animal)) + geom_hline(yintercept=0.902857, linetype='dashed') + scale_color_viridis(discrete=TRUE) + scale_x_log10(breaks=1*10^(-5:1), limits=c(1e-5,NA)) + 
    labs(x=expression(alpha[animal]), y="Accuracy")
gg2 <- gg1 + geom_point(aes(x=alpha_experiment)) + labs(x=expression(alpha[experiment]), y="Accuracy")

gg3 <- ggplot(data_mult, aes(y=accuracy, color=k, shape=dist_chem)) + 
    geom_point(aes(x=alpha_animal)) + geom_hline(yintercept=0.7401428, linetype='dashed') + scale_color_viridis(discrete=TRUE) + scale_x_log10(breaks=1*10^(-5:1), limits=c(1e-5,NA)) + 
    labs(x=expression(alpha[animal]), y="Accuracy")
gg4 <- gg3 + geom_point(aes(x=alpha_experiment)) + labs(x=expression(alpha[experiment]), y="Accuracy")

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Results of the KNN optimization"),
    # Plots for binary classification
    fluidRow(
        column(width = 8, class = "well",
               h4("Binary class accuracy vs. animal feature weights"),
               h5("Left plot controls right plot"),
               fluidRow(
                   column(width = 6,
                          plotOutput("plot1", height = 300,
                                     brush = brushOpts(
                                         id = "plot1_brush",
                                         resetOnNew = TRUE
                                     )
                          )
                   ),
                   column(width = 6,
                          plotOutput("plot2", height = 300)
                   )
               )
        )
    ),
    fluidRow(
        column(width = 8, class = "well",
               h4("Binary class accuracy vs. experiment feature weights"),
               h5("Left plot controls right plot"),
               fluidRow(
                   column(width = 6,
                          plotOutput("plot3", height = 300,
                                     brush = brushOpts(
                                         id = "plot3_brush",
                                         resetOnNew = TRUE
                                     )
                          )
                   ),
                   column(width = 6,
                          plotOutput("plot4", height = 300)
                   )
               )
        )
    ),
    # Plots for multiclass classification
    fluidRow(
        column(width = 8, class = "well",
               h4("Multi class accuracy vs. animal feature weights"),
               h5("Left plot controls right plot"),
               fluidRow(
                   column(width = 6,
                          plotOutput("plot5", height = 300,
                                     brush = brushOpts(
                                         id = "plot5_brush",
                                         resetOnNew = TRUE
                                     )
                          )
                   ),
                   column(width = 6,
                          plotOutput("plot6", height = 300)
                   )
               )
        )
    ),
    fluidRow(
        column(width = 8, class = "well",
               h4("Multi class accuracy vs. experiment feature weights"),
               h5("Left plot controls right plot"),
               fluidRow(
                   column(width = 6,
                          plotOutput("plot7", height = 300,
                                     brush = brushOpts(
                                         id = "plot7_brush",
                                         resetOnNew = TRUE
                                     )
                          )
                   ),
                   column(width = 6,
                          plotOutput("plot8", height = 300)
                   )
               )
        )
    )

)

# Define server logic
server <- function(input, output) {
    
    # -------------------------------------------------------------------
    # Binary classification
    ranges1 <- reactiveValues(x = NULL, y = NULL)
    
    output$plot1 <- renderPlot({gg1})
    
    output$plot2 <- renderPlot({
        gg1 + coord_cartesian(xlim = ranges1$x, ylim = ranges1$y, expand = FALSE)
    })
    
    ranges3 <- reactiveValues(x = NULL, y = NULL)
    
    output$plot3 <- renderPlot({gg2})
    
    output$plot4 <- renderPlot({
        gg2 + coord_cartesian(xlim = ranges3$x, ylim = ranges3$y, expand = FALSE)
    })
    
    # -------------------------------------------------------------------
    # Multiclass classification
    ranges5 <- reactiveValues(x = NULL, y = NULL)
    
    output$plot5 <- renderPlot({gg3})
    
    output$plot6 <- renderPlot({
        gg3 + coord_cartesian(xlim = ranges5$x, ylim = ranges5$y, expand = FALSE)
    })
    
    ranges7 <- reactiveValues(x = NULL, y = NULL)
    
    output$plot7 <- renderPlot({gg4})
    
    output$plot8 <- renderPlot({
        gg4 + coord_cartesian(xlim = ranges7$x, ylim = ranges7$y, expand = FALSE)
    })
    
    # When a double-click happens, check if there's a brush on the plot.
    # If so, zoom to the brush bounds; if not, reset the zoom.
    observe({
        # Binary classification
        brush1 <- input$plot1_brush
        if (!is.null(brush1)) {
            ranges1$x <- c(brush1$xmin, brush1$xmax)
            ranges1$y <- c(brush1$ymin, brush1$ymax)
            
        } else {
            ranges1$x <- NULL
            ranges1$y <- NULL
        }
        brush3 <- input$plot3_brush
        if (!is.null(brush3)) {
            ranges3$x <- c(brush3$xmin, brush3$xmax)
            ranges3$y <- c(brush3$ymin, brush3$ymax)
            
        } else {
            ranges3$x <- NULL
            ranges3$y <- NULL
        }
        # Multiclass classification
        brush5 <- input$plot5_brush
        if (!is.null(brush5)) {
            ranges5$x <- c(brush5$xmin, brush5$xmax)
            ranges5$y <- c(brush5$ymin, brush5$ymax)
            
        } else {
            ranges5$x <- NULL
            ranges5$y <- NULL
        }
        brush7 <- input$plot7_brush
        if (!is.null(brush7)) {
            ranges7$x <- c(brush7$xmin, brush7$xmax)
            ranges7$y <- c(brush7$ymin, brush7$ymax)
            
        } else {
            ranges7$x <- NULL
            ranges7$y <- NULL
        }
    })
    
}

shinyApp(ui, server)