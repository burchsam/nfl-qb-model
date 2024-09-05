library(tidyverse)
library(nflreadr)
library(glmnet)
library(zoo)
library(ggrepel)
library(kknn)
library(randomForestSRC)
library(xgboost)
library(lmtest)
library(car)
library(gt)




rosters = load_rosters(2024)

nfl_qbs_season = read_csv("coding-projects/nfl-qbs/qb-performances-06-23.csv")[, -1]

qbs_pct = nfl_qbs_season |> select(player, qbr_pct_tot, plays_tot) |> unique()




# Searching ---------------------------------------------------------------

rosters |> filter(position == "QB") |> 
  select(team, full_name) |> 
  left_join(qbs_pct, by = c("full_name" = "player")) |> 
  filter(!is.na(qbr_pct_tot)) |>
  mutate(qbr_pct_tot = round(qbr_pct_tot, 2),
         plays_tot = round(plays_tot, -2)) |>
  arrange(team, -qbr_pct_tot) |> 
  print(n = 75)


nfl_qbs_season |> filter(player == "Josh Allen")


nfl_qbs_season |> 
  select(player, qbr_pct_tot) |>
  unique()


nfl_qbs_season |> 
  filter(player == "Trevor Lawrence") |>
  arrange(qbr_pct)

nfl_qbs_season |> 
  group_by(player) |> 
  summarise(season = min(year),
            rookie_play = qbr_pct[year == min(year)],
            dropbacks = dropbacks[year == min(year)],
            .groups = "drop") |> 
  select(season, player, everything()) |> 
  arrange(-rookie_play) |> 
  View()


# Comparing predictors of QB seasonal play --------------------------------

# n vs. n+1

## Composite - 0.53
nfl_qbs_season |> 
  filter(!is.na(qbr_pct)) |> 
  select(player, year, qbr_pct) |> 
  arrange(player, year) |> 
  mutate(next_play = case_when(
    (lead(player) != player) ~ NA,
    .default = lead(qbr_pct)
    )) |> 
  filter(!is.na(next_play)) |> 
  select(next_play, qbr_pct) |> 
  cor()


## EPA/Play - 0.45
nfl_qbs_season |> 
  filter(!is.na(qbr_pct)) |>
  select(player, year, mean_epa) |> 
  arrange(player, year) |> 
  mutate(prev_play = case_when(
    (lag(player) != player) ~ NA,
    .default = lag(mean_epa)
  )) |>
  filter(!is.na(prev_play)) |>
  select(prev_play, mean_epa) |>
  cor()


## Grade - 0.54
nfl_qbs_season |> 
  filter(!is.na(qbr_pct)) |>
  select(player, year, grades_offense) |> 
  arrange(player, year) |> 
  mutate(prev_play = case_when(
    (lag(player) != player) ~ NA,
    .default = lag(grades_offense)
  )) |>
  filter(!is.na(prev_play)) |>
  select(prev_play, grades_offense) |>
  cor()

## CPOE - 0.44
nfl_qbs_season |> 
  filter(!is.na(qbr_pct)) |> 
  select(player, year, cpoe) |> 
  arrange(player, year) |> 
  mutate(next_play = case_when(
    (lead(player) != player) ~ NA,
    .default = lead(cpoe)
  )) |> 
  filter(!is.na(next_play)) |> 
  select(next_play, cpoe) |> 
  cor()


## PTSR - 0.37
nfl_qbs_season |> 
  filter(!is.na(qbr_pct)) |> 
  select(player, year, pressure_to_sack_rate) |> 
  arrange(player, year) |> 
  mutate(next_play = case_when(
    (lead(player) != player) ~ NA,
    .default = lead(pressure_to_sack_rate)
  )) |> 
  filter(!is.na(next_play)) |> 
  select(next_play, pressure_to_sack_rate) |> 
  cor()




# Building the Model ------------------------------------------------------



next_year_df1 = nfl_qbs_season |>
  filter(!is.na(qbr_pct)) |> 
  group_by(player) |> 
  arrange(year) |> 
  # Removes knowledge of next season
  mutate(qbr_pct_avg = cummean(qbr_pct),
         mean_epa_avg = cummean(mean_epa),
         grade_avg = cummean(grades_offense),
         cpoe_avg = cummean(cpoe),
         ptsr_avg = cummean(pressure_to_sack_rate),
         
         qbr_pct_max = cummax(qbr_pct),
         mean_epa_max = cummax(mean_epa),
         grade_max = cummax(grades_offense),
         cpoe_max = cummax(cpoe),
         ptsr_max = cummax(pressure_to_sack_rate),
         dropbacks_max = cummax(dropbacks),
         
         qbr_pct_min = cummin(qbr_pct),
         mean_epa_min = cummin(mean_epa),
         grade_min = cummin(grades_offense),
         cpoe_min = cummin(cpoe),
         ptsr_min = cummin(pressure_to_sack_rate),
         dropbacks_min = cummin(dropbacks),
         
         first_year = min(year),
         qbr_pct_first = qbr_pct[year == first_year],
         mean_epa_first = mean_epa[year == first_year],
         grade_first = grades_offense[year == first_year],
         cpoe_first = cpoe[year == first_year],
         ptsr_first = pressure_to_sack_rate[year == first_year],
         dropbacks_first = dropbacks[year == first_year],
         
         # Add median year other pcts?
         
         plays_tot = cumsum(dropbacks)) |>   
  # left_join(rookie_df, by = "player") |> 
  mutate(next_qbr_pct = case_when(
    (lead(player) == player) ~ lead(qbr_pct),
    .default = NA
  )) |>
  filter(!is.na(next_qbr_pct)) |>
  ungroup() |>
  select(-c(player, year, qbr_pct_tot, mean_epa_tot, cpoe_tot, ptsr_tot, first_year))



next_year_df1 |> View()


# ggplot(next_year_df1, aes(x = plays_tot, y = next_qbr_pct)) +
#   geom_point() +
#   stat_smooth(formula = y ~ x, geom = 'line', se = FALSE, color='blue') +
#   theme_minimal()

cor_tbl = cor(next_year_df1) |> 
  as_tibble() |> 
  mutate(names = colnames(next_year_df1)) |> 
  filter(next_qbr_pct != 1) |> 
  select(names, next_qbr_pct) |> 
  arrange(-abs(next_qbr_pct))


ggplot(cor_tbl, aes(x = abs(next_qbr_pct), y = reorder(names, abs(next_qbr_pct)))) +
  geom_col(aes(fill = next_qbr_pct > 0), alpha = .8) +
  labs(
    title = "Absolute Correlation with Next Year's QB Percentile",
    subtitle = "blue = positive  |  red = negative  |  percentile = 50-50 composite of epa/play & pff grade",
    y = "Predictors",
    x = "Correlation",
    caption = "By: Sam Burch  |  Data: nflfastR & pff (2006-2023)",
    fill = element_blank()
  ) +
  scale_x_continuous(breaks = seq(0, 1, .1)) +
  scale_fill_brewer(palette = "Set1") + 
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, size = 6),
        plot.caption = element_text(size = 6),
        axis.line = element_line(color = "black", size = 0.5),
        panel.grid.major.x = element_line(color = "lightgray", size = 0.5, linetype = 2),  # Customize vertical major grid lines
        panel.grid = element_blank(),
        panel.background = element_blank(),
        legend.position = "none")

# ggsave("qb-pct-correlations.png", width = 16, height = 12, units = "cm")




# Train/Test

n = nrow(next_year_df1)



set.seed(123)

train_ind = sample(1:n, .7*n)

train_data = next_year_df1[train_ind, ]
test_data = next_year_df1[-train_ind, ]


next_year_df1 |> select(next_qbr_pct) |> pull() |> mean()
## Average is 0.56th, higher than 50th


# PCA


pca_result = prcomp(train_data |> select(-next_qbr_pct), scale. = TRUE)


plot(pca_result)


pca_data = data.frame(
  PC1 = pca_result$x[,1],
  PC2 = pca_result$x[,2], 
  Play = train_data |> select(next_qbr_pct) |> pull())

# Plot the data on the first two principal components, colored by the digit label
ggplot(pca_data, aes(x = PC1, y = Play)) +
  geom_point() +
  labs(
    # title = "PCA of MNIST Data (Digits 1 and 2)",
    x = "First Principal Component",
    y = "Second Principal Component",
    color = "Ranking"
  ) +
  theme_minimal()
## Not good enough explanation

pca_data |> select(PC1, Play) |> cor()


# KMeans

kmeans_result = kmeans(train_data |> select(-next_qbr_pct), centers = 5)

kmeans_result$withinss




# Linear Regression


## Model 1
m1 = lm(next_qbr_pct ~ ., data = train_data)
summary(m1)
# r^2 = 0.32

pred = predict(m1, newdata = test_data)

sqrt(mean((pred - test_data |> select(next_qbr_pct) |> pull())^2))
# Off by 0.234



## Model 2
m2 = step(m1, trace = 0)
summary(m2)
# r^2 = 0.34

pred = predict(m2, newdata = test_data)

sqrt(mean((pred - test_data |> select(next_qbr_pct) |> pull())^2))
# Off by 0.235

sqrt(vif(m2))

par(mfrow = c(2, 2))
plot(m2)
# Diagnostics NOT ok!



# Model 3 (Last)
m3 = lm(next_qbr_pct ~ ., data = train_data |> select(qbr_pct:dropbacks, next_qbr_pct))
summary(m3)
# r^2 = 0.28

pred = predict(m3, newdata = test_data)

sqrt(mean((pred - test_data |> select(next_qbr_pct) |> pull())^2))
# Off by 0.236

m4 = step(m3, trace = 0)
summary(m4)
# r^2 = 0.29

pred = predict(m4, newdata = test_data)

sqrt(mean((pred - test_data |> select(next_qbr_pct) |> pull())^2))
# Off by 0.236

sqrt(vif(m4))

par(mfrow = c(2, 2))
plot(m4)
# Diagnostics ok!




# m4 = lm(next_qbr_pct ~ ., data = train_data |> select(mean_epa_tot:plays_tot, next_qbr_pct))
# summary(m4)
# 
# pred = predict(m4, newdata = test_data)
# 
# sqrt(mean((pred - test_data |> select(next_qbr_pct) |> pull())^2))
# # Off by 0.240
# 
# sqrt(vif(m4))
# 
# 
# m5 = step(m4, trace = 0)
# summary(m5)
# 
# pred = predict(m5, newdata = test_data)
# 
# sqrt(mean((pred - test_data |> select(next_qbr_pct) |> pull())^2))
# # 0.241
# 
# sqrt(vif(m5))



## LASSO

set.seed(123)

x_train = as.matrix(train_data |> select(-next_qbr_pct))
y_train = as.matrix(train_data$next_qbr_pct)

x_test = as.matrix(test_data |> select(-next_qbr_pct))
y_test = as.matrix(test_data$next_qbr_pct)


lasso_model = glmnet(x_train, y_train, alpha = 1)


# Perform cross-validation to select lambda
cv_lasso = cv.glmnet(x_train, y_train, alpha = 1)  # alpha = 1 for Lasso regression
par(mfrow = c(1,1))
plot(cv_lasso)

# Print optimal lambda value
print(cv_lasso$lambda.min)

coefficients = coef(lasso_model, s = cv_lasso$lambda.min)
coefficients

# Example predictions
pred = predict(lasso_model, newx = x_test, s = cv_lasso$lambda.min)

sqrt(mean((pred - y_test)^2))
# Off by .233


# RIDGE
set.seed(123)


ridge_model = cv.glmnet(x_train, y_train, alpha = 0)

plot(ridge_model)

best_lambda = ridge_model$lambda.min

final_model = glmnet(x_train, y_train, alpha = 0, lambda = best_lambda)

final_model$beta[, 1] |> abs() |> sort(decreasing = TRUE)

pred = predict(final_model, newx = x_test)

sqrt(mean((pred - y_test)^2))
# RMSE of .233
# Reduced collinearity effects



# KNN

error = numeric(100)

set.seed(123)

for (i in 1:100) {
  
  knn.fit = kknn(next_qbr_pct ~ ., train = train_data, 
                 test = test_data |> dplyr::select(-next_qbr_pct),
                 k = i, kernel = "rectangular")
  
  test.pred = knn.fit$fitted.values
  
  error[i] = sqrt(mean((test.pred - (test_data |> dplyr::select(next_qbr_pct) |> pull()))^2))
  
  
}

min(error)
## 0.232

which.min(error)




# Random Forrest


set.seed(123)


# train_data$qbr_nfl
m1 = rfsrc(next_qbr_pct ~ ., data = as.data.frame(train_data))
# OOB Error Rate
tail(m1$err.rate, 1)


tuning_grid = expand.grid(mtry = c(1, 5, 10, 15, 20), nodesize = c(1, 5, 10, 15, 20))
tuned_models = vector(mode = "list", length = 25)
oob_error_rates = numeric(25)
set.seed(123)
for (i in 1:nrow(tuning_grid)) {
  rf_model = rfsrc(next_qbr_pct ~ ., data = as.data.frame(train_data),
                   mtry = tuning_grid[i, 1], nodesize = tuning_grid[i, 2])
  tuned_models[[i]] = rf_model
  oob_error_rates[i] = tail(rf_model$err.rate, 1)
}
# OOB ER for each model
oob_error_rates


# Find the index of the minimum OOB error rate
best_index = which.min(oob_error_rates)
# Best tuning parameters
best_tuning = tuning_grid[best_index, ]
best_tuning

# Extract the random forest model with the best tuning parameters
best_rf_model = tuned_models[[best_index]]
best_rf_model
# r^2 = 0.32

# Calculate the variable importance for the best model
variable_importance = vimp(best_rf_model)
sort(abs(variable_importance$importance), decreasing = TRUE)



rf_weights = variable_importance$importance |> 
  as_tibble() |> 
  rename(weight = value) |> 
  mutate(names = variable_importance$xvar.names) |> 
  select(names, weight) |> 
  arrange(-weight)

ggplot(rf_weights, aes(x = weight, y = reorder(names, weight))) +
  geom_col() +
  theme_minimal()


pred_rf = predict(best_rf_model, newdata = as.data.frame(test_data))

sqrt(mean((pred_rf$predicted - as.vector(y_test))^2))
## 0.226




### XGBoost

x_train = as.matrix(train_data |> select(-next_qbr_pct))
y_train = as.matrix(train_data$next_qbr_pct)
x_test = as.matrix(test_data |> select(-next_qbr_pct))
y_test = as.matrix(test_data$next_qbr_pct)



set.seed(123)

train_data_xgb = xgb.DMatrix(data = data.matrix(train_data |> select(-next_qbr_pct)), label = train_data$next_qbr_pct)
test_data_xgb = xgb.DMatrix(data = data.matrix(test_data |> select(-next_qbr_pct)), label = test_data$next_qbr_pct)

params = list(
  objective = "reg:squarederror",
  # num_class = 10, # Number of classes
  eta = 0.5, # Learning rate
  max_depth = 2 # Maximum depth of trees
)

num_round = 50


xgb.fit = xgb.train(params, train_data_xgb, num_round)


pred_xgb = predict(xgb.fit, newdata = test_data_xgb)

sqrt(mean((pred_xgb - as.vector(y_test))^2))
## Not awful (0.255)




tuning_grid2 = expand.grid(eta = c(0.1, 0.5, 1.0), max_depth = c(2, 5, 10))


best_eta = c(numeric(9))
best_max_depth = c(numeric(9))
best_ntrees = c(numeric(9))
best_error = c(rep(1, 9))

set.seed(123)

# Loop over the search space
for (i in 1:nrow(tuning_grid2)) {
  # Set the xgboost parameters
  params = list(
    objective = "reg:squarederror",
    # num_class = 10,
    eta = tuning_grid2[i, 1],
    max_depth = tuning_grid2[i, 2]
  )
  
  train_data_xgb = xgb.DMatrix(data = data.matrix(train_data |> select(-next_qbr_pct)), label = train_data$next_qbr_pct)
  test_data_xgb = xgb.DMatrix(data = data.matrix(test_data |> select(-next_qbr_pct)), label = test_data$next_qbr_pct)
  
  num_round = 50
  
  bst = xgb.train(params, train_data_xgb, num_round)
  
  for (j in 1:50) {
    pred_xgb = predict(bst, test_data_xgb, iterationrange = c(1, j))
    
    # sqrt(mean((pred_xgb - as.vector(y_test))^2))
    # Calculate the testing error
    error = sqrt(mean((pred_xgb - as.vector(y_test))^2))
    
    # Update the best parameters and the corresponding testing error
    if (error < best_error[i]) {
      best_eta[i] = tuning_grid2[i, 1]
      best_max_depth[i] = tuning_grid2[i, 2]
      best_ntrees[i] = j
      best_error[i] = error
    }
    
  }
}

(results = data.frame(best_eta, best_max_depth, best_ntrees, best_error))

# Best Error
results[which.min(best_error), ]
## 0.227





# Predictions! ------------------------------------------------------------


qbs_2023 = nfl_qbs_season |>
  filter(!is.na(qbr_pct)
         # , !is.na(qbr_pct_tot)
         ) |> 
  group_by(player) |> 
  arrange(year) |> 
  mutate(qbr_pct_avg = cummean(qbr_pct),
         mean_epa_avg = cummean(mean_epa),
         grade_avg = cummean(grades_offense),
         cpoe_avg = cummean(cpoe),
         ptsr_avg = cummean(pressure_to_sack_rate),
         
         qbr_pct_max = cummax(qbr_pct),
         mean_epa_max = cummax(mean_epa),
         grade_max = cummax(grades_offense),
         cpoe_max = cummax(cpoe),
         ptsr_max = cummax(pressure_to_sack_rate),
         dropbacks_max = cummax(dropbacks),
         
         qbr_pct_min = cummin(qbr_pct),
         mean_epa_min = cummin(mean_epa),
         grade_min = cummin(grades_offense),
         cpoe_min = cummin(cpoe),
         ptsr_min = cummin(pressure_to_sack_rate),
         dropbacks_min = cummin(dropbacks),
         
         first_year = min(year),
         qbr_pct_first = qbr_pct[year == first_year],
         mean_epa_first = mean_epa[year == first_year],
         grade_first = grades_offense[year == first_year],
         cpoe_first = cpoe[year == first_year],
         ptsr_first = pressure_to_sack_rate[year == first_year],
         dropbacks_first = dropbacks[year == first_year],
         
         plays_tot = cumsum(dropbacks)) |> 
  ungroup() |> 
  filter((year == 2023 | (player == "Aaron Rodgers" & year == 2022) |
                        (player == "Sam Darnold" & year == 2022) |
                        (player == "Jacoby Brissett" & year == 2022) |
                        (player == "Marcus Mariota"& year == 2022) |
                        (player == "Drew Lock" & year == 2020)) &
                        (!is.na(qbr_pct_tot) | player == "Will Levis" | player == "Aidan O'Connell")) |>
  arrange(-qbr_pct_tot)

qbs_2023 |> View()





# pred_2024 = predict(xgb.fit, newdata = xgb.DMatrix(as.matrix(qbs_2023 |> select(-c(player, year, qbr_pct_tot, mean_epa_tot, cpoe_tot, ptsr_tot)))))

pred_2024 = predict(best_rf_model, newdata = as.data.frame(qbs_2023))
pred_2024$predicted

qb_teams = rosters |> filter(position == "QB") |> select(full_name, team)


qbs_2024 = qbs_2023 |>
  mutate(pred_24 = pred_2024$predicted) |> 
  select(player, pred_24, qbr_pct_tot, mean_epa_tot, grade_avg, everything()) |> 
  arrange(-pred_24) |> 
  left_join(qb_teams, by = c("player" = "full_name")) |> 
  filter(!is.na(team))

qbs_2024 |> 
  View()



  
# Projections vs. Career
ggplot(qbs_2024 |>  filter(!is.na(qbr_pct_tot)), aes(x = qbr_pct_tot, y = pred_24)) +
  # geom_point(aes(color = team, fill = team)) +
  stat_smooth(formula = y ~ x, method = 'lm', geom = 'line', se = FALSE, color='gray') +
  geom_text_repel(aes(label = player), size = 1.5) +
  labs(
    title = "Career QB Play vs. 2024 Projections",
    subtitle = "23' dropbacks >= 150, Rodgers, Darnold, Brissett, Mariota, or Lock  |  career dropbacks >= 400  |  percentile = 50-50 compositie of epa/play & pff grade  |  projections use random forest model",
    caption = "By: Sam Burch  |  Data @nflfastR & @pff (since 2006)",
    x = "Career Percentile",
    y = "2024 Percentile Projections"
  ) +
  # nflplotR::scale_fill_nfl(alpha = .8) +
  # nflplotR::scale_color_nfl(type = "primary") +
  nflplotR::geom_mean_lines(aes(x0 = qbr_pct_tot, y0 = pred_24, alpha = .8)) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    legend.position = "none",
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 4.5),
    plot.caption = element_text(size = 7),
    axis.line = element_line(color = "black", size = 0.5)
  ) +
  nflplotR::geom_nfl_logos(aes(team_abbr = team), width = .03, alpha = .8)

# ggsave("nfl-qbs-24-career.png", width = 16, height = 9, units = "cm")





# Young QBs
qbs_2023 |> 
  filter(plays_tot <= 1500) |> 
  select(player, qbr_pct_tot:plays_tot) |> 
  arrange(-grade_avg)


ggplot(qbs_2024 |> filter(!is.na(qbr_pct_tot)), aes(x = grade_avg, y = mean_epa_tot)) +
  geom_point(alpha = .3, color = "grey20") +
  geom_point(aes(color = team, fill = team), data = qbs_2024 |> filter(plays_tot <= 1500)) +
  stat_smooth(formula = y ~ x, method = 'lm', geom = 'line', se = FALSE, color='gray') +
  geom_text_repel(aes(label = player), size = 2, data = qbs_2024 |> filter(plays_tot <= 1500)) +
  # scale_x_continuous(breaks = seq(0, 120, 10)) +
  # scale_y_continuous(breaks = seq(0, 120, 15)) +
  labs(
    title = "Young QB Career Performances",
    subtitle = "400-1500 career dropbacks  |  150+ single season dropbacks",
    caption = "By: Sam Burch  |  Data @nflfastR & @pff (since 2006)",
    x = "Average PFF Grade",
    y = "EPA / play"
  ) +
  nflplotR::scale_fill_nfl(alpha = .8) +
  nflplotR::scale_color_nfl(type = "primary") +
  nflplotR::geom_mean_lines(aes(x0 = grade_avg, y0 = mean_epa_tot, alpha = .8)) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    legend.position = "none",
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, size = 5),
    plot.caption = element_text(size = 7),
    axis.line = element_line(color = "black", size = 0.5)
  )

# ggsave("nfl-qbs-young-careers.png", width = 16, height = 9, units = "cm")





# 2024 Potential Starters
## Plus 6 rookies & Richardson

qbs_24_red = qbs_2024 |> 
  filter(
    player != "Taylor Heinicke",
    player != "Nick Mullens",
    player != "Mac Jones",
    player != "Desmond Ridder",
    player != "Jimmy Garoppolo",
    player != "Sam Howell",
    player != "Joshua Dobbs",
    player != "Tyrod Taylor",
    player != "Kenny Pickett",
    player != "Joe Flacco"
  )

gt_theme_espn = function(data, ...) {
  data |> 
    opt_all_caps()  |> 
    opt_table_font(
      font = list(
        google_font("Lato"),
        default_fonts()
      )
    )  |> 
    opt_row_striping() |> 
    tab_options(
      row.striping.background_color = "#fafafa",
      table_body.hlines.color = "#f6f7f7",
      source_notes.font.size = 12,
      table.font.size = 16,
      table.width = px(700),
      heading.align = "left",
      heading.title.font.size = 24,
      table.border.top.color = "transparent",
      table.border.top.width = px(3),
      data_row.padding = px(7),
      ...
    ) 
}


qbs_24_red |> 
  select(team, player, pred_24, qbr_pct, qbr_pct_tot) |> 
  mutate(round(across(pred_24:qbr_pct_tot), 2)) |> 
  gt() |> 
  gt_theme_espn()



# Old ---------------------------------------------------------------------



# next_year_df1 = nfl_qbs_season |> 
#   select(player:dropbacks) |> 
#   filter(!is.na(qbr_pct)) |> 
#   arrange(player, year) |> 
#   mutate(qbr_pct_5 = case_when(
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) & (lag(lag(lag(lag(player)))) == player) ~ rollmean(qbr_pct, k = 5, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) & (lag(lag(lag(lag(player)))) != player) ~ rollmean(qbr_pct, k = 4, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) != player) ~ rollmean(qbr_pct, k = 3, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) != player) ~ rollmean(qbr_pct, k = 2, fill = NA),
#     .default = qbr_pct
#          ),
#          qbr_pct_4 = case_when(
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) ~ rollmean(qbr_pct, k = 4, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) != player) ~ rollmean(qbr_pct, k = 3, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) != player) ~ rollmean(qbr_pct, k = 2, fill = NA),
#     .default = qbr_pct
#          ),
#          qbr_pct_3 = case_when(
#     (lag(player) == player) & (lag(lag(player)) == player) ~ rollmean(qbr_pct, k = 3, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) != player) ~ rollmean(qbr_pct, k = 2, fill = NA),
#     .default = qbr_pct
#          ),
#          qbr_pct_2 = case_when(
#     (lag(player) == player) ~ rollmean(qbr_pct, k = 2, fill = NA),
#     .default = qbr_pct
#          )) |>
#   mutate(mean_epa_5 = case_when(
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) & (lag(lag(lag(lag(player)))) == player) ~ rollmean(mean_epa, k = 5, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) & (lag(lag(lag(lag(player)))) != player) ~ rollmean(mean_epa, k = 4, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) != player) ~ rollmean(mean_epa, k = 3, fill = NA),
#     (lag(player) == player) ~ rollmean(mean_epa, k = 2, fill = NA),
#     .default = mean_epa
#         ),
#     mean_epa_4 = case_when(
#       (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) ~ rollmean(mean_epa, k = 4, fill = NA),
#       (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) != player) ~ rollmean(mean_epa, k = 3, fill = NA),
#       (lag(player) == player) & (lag(lag(player)) != player) ~ rollmean(mean_epa, k = 2, fill = NA),
#       .default = mean_epa
#     ),
#     mean_epa_3 = case_when(
#       (lag(player) == player) & (lag(lag(player)) == player) ~ rollmean(mean_epa, k = 3, fill = NA),
#       (lag(player) == player) & (lag(lag(player)) != player) ~ rollmean(mean_epa, k = 2, fill = NA),
#       .default = mean_epa
#     ),
#     mean_epa_2 = case_when(
#       (lag(player) == player) ~ rollmean(mean_epa, k = 2, fill = NA),
#       .default = mean_epa
#     )) |>
#   mutate(grades_5 = case_when(
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) & (lag(lag(lag(lag(player)))) == player) ~ rollmean(grades_offense, k = 5, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) & (lag(lag(lag(lag(player)))) != player) ~ rollmean(grades_offense, k = 4, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) != player) ~ rollmean(grades_offense, k = 3, fill = NA),
#     (lag(player) == player) ~ rollmean(grades_offense, k = 2, fill = NA),
#     .default = grades_offense
#         ),
#     grades_4 = case_when(
#       (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) ~ rollmean(grades_offense, k = 4, fill = NA),
#       (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) != player) ~ rollmean(grades_offense, k = 3, fill = NA),
#       (lag(player) == player) & (lag(lag(player)) != player) ~ rollmean(grades_offense, k = 2, fill = NA),
#       .default = grades_offense
#     ),
#     grades_3 = case_when(
#       (lag(player) == player) & (lag(lag(player)) == player) ~ rollmean(grades_offense, k = 3, fill = NA),
#       (lag(player) == player) & (lag(lag(player)) != player) ~ rollmean(grades_offense, k = 2, fill = NA),
#       .default = grades_offense
#     ),
#     grades_2 = case_when(
#       (lag(player) == player) ~ rollmean(grades_offense, k = 2, fill = NA),
#       .default = grades_offense
#     )) |>
#   mutate(cpoe_5 = case_when(
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) & (lag(lag(lag(lag(player)))) == player) ~ rollmean(cpoe, k = 5, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) & (lag(lag(lag(lag(player)))) != player) ~ rollmean(cpoe, k = 4, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) != player) ~ rollmean(cpoe, k = 3, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) != player) ~ rollmean(cpoe, k = 2, fill = NA),
#     .default = cpoe
#   ),
#   cpoe_4 = case_when(
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) ~ rollmean(cpoe, k = 4, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) != player) ~ rollmean(cpoe, k = 3, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) != player) ~ rollmean(cpoe, k = 2, fill = NA),
#     .default = cpoe
#   ),
#   cpoe_3 = case_when(
#     (lag(player) == player) & (lag(lag(player)) == player) ~ rollmean(cpoe, k = 3, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) != player) ~ rollmean(cpoe, k = 2, fill = NA),
#     .default = cpoe
#   ),
#   cpoe_2 = case_when(
#     (lag(player) == player) ~ rollmean(cpoe, k = 2, fill = NA),
#     .default = cpoe
#   )) |>
#   mutate(ptsr_5 = case_when(
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) & (lag(lag(lag(lag(player)))) == player) ~ rollmean(pressure_to_sack_rate, k = 5, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) & (lag(lag(lag(lag(player)))) != player) ~ rollmean(pressure_to_sack_rate, k = 4, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) != player) ~ rollmean(pressure_to_sack_rate, k = 3, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) != player) ~ rollmean(pressure_to_sack_rate, k = 2, fill = NA),
#     .default = pressure_to_sack_rate
#   ),
#   ptsr_4 = case_when(
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) ~ rollmean(pressure_to_sack_rate, k = 4, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) != player) ~ rollmean(pressure_to_sack_rate, k = 3, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) != player) ~ rollmean(pressure_to_sack_rate, k = 2, fill = NA),
#     .default = pressure_to_sack_rate
#   ),
#   ptsr_3 = case_when(
#     (lag(player) == player) & (lag(lag(player)) == player) ~ rollmean(pressure_to_sack_rate, k = 3, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) != player) ~ rollmean(pressure_to_sack_rate, k = 2, fill = NA),
#     .default = pressure_to_sack_rate
#   ),
#   ptsr_2 = case_when(
#     (lag(player) == player) ~ rollmean(pressure_to_sack_rate, k = 2, fill = NA),
#     .default = pressure_to_sack_rate
#   )) |>
#   mutate(dropbacks_5 = case_when(
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) & (lag(lag(lag(lag(player)))) == player) ~ rollsum(dropbacks, k = 5, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) & (lag(lag(lag(lag(player)))) != player) ~ rollsum(dropbacks, k = 4, fill = NA),
#     (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) != player) ~ rollsum(dropbacks, k = 3, fill = NA),
#     (lag(player) == player) ~ rollsum(dropbacks, k = 2, fill = NA),
#     .default = dropbacks
#         ),
#     dropbacks_4 = case_when(
#       (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) == player) ~ rollsum(dropbacks, k = 4, fill = NA),
#       (lag(player) == player) & (lag(lag(player)) == player) & (lag(lag(lag(player))) != player) ~ rollsum(dropbacks, k = 3, fill = NA),
#       (lag(player) == player) & (lag(lag(player)) != player) ~ rollsum(dropbacks, k = 2, fill = NA),
#       .default = dropbacks
#     ),
#     dropbacks_3 = case_when(
#       (lag(player) == player) & (lag(lag(player)) == player) ~ rollsum(dropbacks, k = 3, fill = NA),
#       (lag(player) == player) & (lag(lag(player)) != player) ~ rollsum(dropbacks, k = 2, fill = NA),
#       .default = dropbacks
#     ),
#     dropbacks_2 = case_when(
#       (lag(player) == player) ~ rollsum(dropbacks, k = 2, fill = NA),
#       .default = dropbacks
#     )) |>
#   mutate(next_qbr_pct = case_when(
#     (lead(player) == player) ~ lead(qbr_pct),
#     .default = NA
#   )) |> 
#   filter(!is.na(next_qbr_pct)) |> 
#   select(-player, -year)




