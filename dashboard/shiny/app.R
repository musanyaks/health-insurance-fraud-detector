# Investigation dashboard - 4 tabs: Queue / Claim detail / Providers / Model
library(shiny)
library(DT)
library(plotly)
library(jsonlite)

queue   <- read.csv("data/processed/investigation_queue.csv", stringsAsFactors = FALSE)
metrics <- fromJSON("models/xgb_metrics.json")

ui <- navbarPage(
  "Claims Fraud Investigator",
  theme = "cosmo",

  tabPanel("Queue",
    fluidRow(
      column(3, selectInput("risk", "Risk level:", c("ALL", "HIGH", "MEDIUM", "LOW"))),
      column(3, selectInput("ftype", "Planted mechanism:", c("ALL", unique(queue$fraud_type))))
    ),
    DT::dataTableOutput("queue_table")
  ),

  tabPanel("Claim detail",
    fluidRow(
      column(4, selectInput("claim", "Claim:", queue$claim_id)),
      column(8,
        plotlyOutput("gauge", height = "220px"),
        h4("Reasons"),
        verbatimTextOutput("reasons"),
        verbatimTextOutput("claim_meta")
      )
    )
  ),

  tabPanel("Provider insights",
    fluidRow(
      column(6, plotlyOutput("prov_count")),
      column(6, plotlyOutput("prov_amount"))
    )
  ),

  tabPanel("Model",
    h4("Time-split evaluation (train on past, test on future)"),
    verbatimTextOutput("model_metrics"),
    p("Recall @ 10% review = share of fraud caught if investigators only review ",
      "the riskiest 10% of claims. See docs/fraud_taxonomy.md for per-mechanism results.")
  )
)

server <- function(input, output, session) {

  filtered <- reactive({
    q <- queue
    if (input$risk != "ALL")  q <- q[q$risk_level == input$risk, ]
    if (input$ftype != "ALL") q <- q[q$fraud_type == input$ftype, ]
    q[order(-q$fraud_risk_score), ]
  })

  output$queue_table <- DT::renderDataTable({
    DT::datatable(filtered(), rownames = FALSE,
      colnames = c("Claim", "Member", "Provider", "Code", "Date", "Amount",
                   "Actual fraud", "Mechanism", "P(fraud)", "Score", "Level",
                   "Rules", "Reasons", "Action")[1:ncol(queue)],
      options = list(pageLength = 15, scrollX = TRUE)) |>
      DT::formatStyle("risk_level",
        color = DT::styleEqual(c("HIGH", "MEDIUM"), c("#c0392b", "#e67e22")))
  })

  selected <- eventReactive(input$claim, queue[queue$claim_id == input$claim, ][1, ])

  output$gauge <- renderPlotly({
    s <- selected()$fraud_risk_score
    plot_ly(type = "indicator", mode = "gauge+number", value = s,
            gauge = list(axis = list(range = c(0, 100)),
                         bar = list(color = "darkblue"),
                         steps = list(list(range = c(0, 40), color = "#c8e6c9"),
                                      list(range = c(40, 70), color = "#fff3cd"),
                                      list(range = c(70, 100), color = "#f8d7da"))),
            title = list(text = "Fraud risk score"))
  })

  output$reasons <- renderPrint(cat(selected()$reasons))
  output$claim_meta <- renderPrint({
    s <- selected()
    cat(sprintf("Claim %s | %s | %s | KES %,.0f | %s\nML P(fraud): %.3f | Rules: %s",
                s$claim_id, s$provider_id, s$code, s$claim_amount, s$claim_date,
                s$ml_probability, s$rule_hits))
  })

  agg <- reactive({
    aggregate(cbind(n = fraud_risk_score, amount = claim_amount) ~ provider_id,
              data = queue, FUN = function(x) c(count = length(x), total = sum(x)))
  })

  output$prov_count <- renderPlotly({
    a <- agg(); d <- data.frame(provider_id = a$provider_id, n = a$n[, "count"])
    d <- d[order(-d$n), ][1:15, ]
    plot_ly(d, x = ~provider_id, y = ~n, type = "bar") |>
      layout(title = "Top 15 flagged providers (claim count)")
  })

  output$prov_amount <- renderPlotly({
    a <- agg(); d <- data.frame(provider_id = a$provider_id, amt = a$amount[, "total"])
    d <- d[order(-d$amt), ][1:15, ]
    plot_ly(d, x = ~provider_id, y = ~amt, type = "bar") |>
      layout(title = "Top 15 flagged providers (flagged KES)")
  })

  output$model_metrics <- renderPrint({
    cat(sprintf("Model:            %s\nPR-AUC:           %.3f\nRecall @ 10%%:     %.3f\nCutoff date:      %s\nTrain/Test sizes: %d / %d",
                metrics$model, metrics$pr_auc, metrics$recall_at_10pct,
                metrics$cutoff_date, metrics$n_train, metrics$n_test))
  })
}

shinyApp(ui, server)