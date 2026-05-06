# =============================================================================
# Analysis script: How caregivers coordinate deictic gestures with referential
#                  and syntactic structure
#
# Authors: Erjing Zhang, Emily M. Neer, Catherine M. Sandhofer
# Data:    data/socialcue_data.csv
#          21,220 caregiver utterances from 15 children (Wave 1)
#
# This script reproduces the analyses and figures reported in the manuscript.
# Run from the project root. Outputs: ./figures/ and ./tables/.
#
# To reproduce:
#   1. Open the project in RStudio (or set the working directory to the repo
#      root).
#   2. install.packages(c("here", "readr", "dplyr", "lme4", "car",
#                         "broom.mixed", "emmeans", "ggeffects", "ggplot2"))
#   3. source("analysis.R")
# =============================================================================


# ---- Packages ---------------------------------------------------------------
library(here)
library(readr)
library(dplyr)
library(lme4)
library(car)
library(broom.mixed)
library(emmeans)
library(ggeffects)
library(ggplot2)

# Output directories
dir.create(here("figures"), showWarnings = FALSE)
dir.create(here("tables"),  showWarnings = FALSE)


# ---- Load data --------------------------------------------------------------
data <- read_csv(here("data", "socialcue_data.csv"), show_col_types = FALSE)


# ---- Clean and derive variables --------------------------------------------
# Treat missing presence-of-cue/count as 0 (utterance had no recorded cue/word).
# Same_Referent is left as NA where not applicable; it is filtered per-analysis.
data <- data %>%
  mutate(
    Showing       = tidyr::replace_na(as.numeric(Showing),       0),
    Pointing      = tidyr::replace_na(as.numeric(Pointing),      0),
    Noun_count    = tidyr::replace_na(as.numeric(Noun_count),    0),
    Pronoun_count = tidyr::replace_na(as.numeric(Pronoun_count), 0),
    Social_Cue    = ifelse(Showing == 1 | Pointing == 1, 1, 0),
    utterance_type = factor(
      case_when(
        Noun_count > 0  & Pronoun_count > 0  ~ "Pronoun & Noun",
        Noun_count > 0  & Pronoun_count == 0 ~ "Only Noun",
        Noun_count == 0 & Pronoun_count > 0  ~ "Only Pronoun",
        Noun_count == 0 & Pronoun_count == 0 ~ "No Pronoun or Noun"
      ),
      levels = c("Only Pronoun", "Only Noun", "Pronoun & Noun", "No Pronoun or Noun")
    ),
    # Standardize MLU to facilitate interpretation (per Methods).
    # Raw MLU is retained in the column `MLU`; modeling uses MLU_z.
    MLU_z = as.numeric(scale(MLU))
  )


# Shared theme for publication figures
pub_theme <- theme_minimal(base_size = 16) +
  theme(
    plot.title   = element_blank(),
    axis.title   = element_text(size = 18, colour = "black"),
    axis.text    = element_text(size = 16, colour = "black"),
    legend.title = element_text(size = 16, colour = "black"),
    legend.text  = element_text(size = 14, colour = "black"),
    text         = element_text(colour = "black")
  )


# =============================================================================
# Q1. Does gesture probability vary by utterance type?
# =============================================================================

# Chi-square test
cue_table <- table(data$utterance_type, data$Social_Cue)
print(cue_table)
print(chisq.test(cue_table))

# Mixed-effects logistic regression with random intercept by child
m_utt <- glmer(
  Social_Cue ~ utterance_type + (1 | ID),
  data = data, family = binomial
)
summary(m_utt)
Anova(m_utt, type = "III")

# Descriptive proportions with 95% CI for plotting
summary_data <- data %>%
  group_by(utterance_type) %>%
  summarise(
    Prop     = mean(Social_Cue),
    SE       = sqrt((Prop * (1 - Prop)) / n()),
    CI_Lower = pmax(0, Prop - 1.96 * SE),
    CI_Upper = pmin(1, Prop + 1.96 * SE),
    .groups  = "drop"
  )

# Figure: descriptive proportion of gesture by utterance type
fig_utterance_type <- ggplot(summary_data,
                             aes(x = utterance_type, y = Prop, fill = utterance_type)) +
  geom_col(alpha = 0.8) +
  geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), width = 0.2) +
  labs(x = "Utterance Type",
       y = "Proportion of Utterances with Gesture") +
  pub_theme +
  theme(legend.position = "none")
ggsave(here("figures", "fig_utterance_type.png"),
       fig_utterance_type, width = 7, height = 5, dpi = 300)

# Table: model estimates for utterance-type effect
tab_utt <- tidy(m_utt, effects = "fixed", conf.int = TRUE, conf.level = 0.95) %>%
  mutate(
    Predictor = case_when(
      term == "(Intercept)"                      ~ "Intercept (Only Pronoun)",
      term == "utterance_typeOnly Noun"          ~ "Only Noun vs. Only Pronoun",
      term == "utterance_typePronoun & Noun"     ~ "Pronoun + Noun vs. Only Pronoun",
      term == "utterance_typeNo Pronoun or Noun" ~ "Neither vs. Only Pronoun",
      TRUE                                       ~ term
    ),
    p.value = ifelse(p.value < .001, "< .001", sprintf("%.3f", p.value))
  ) %>%
  select(Predictor,
         Estimate       = estimate,
         SE             = std.error,
         `95% CI Lower` = conf.low,
         `95% CI Upper` = conf.high,
         p              = p.value)
write_csv(tab_utt, here("tables", "Table_UtteranceType_Gesture.csv"))

# Figure: model-predicted probability of gesture by utterance type
emm <- emmeans(m_utt, ~ utterance_type, type = "response")
print(summary(emm))
print(pairs(emm, adjust = "bonferroni"))

emm_df <- as.data.frame(emm)
fig_utt_predicted <- ggplot(emm_df,
                            aes(x = utterance_type, y = prob, fill = utterance_type)) +
  geom_col(alpha = 0.8) +
  geom_errorbar(aes(ymin = asymp.LCL, ymax = asymp.UCL), width = 0.2) +
  labs(x = "Utterance Type",
       y = "Predicted Probability of Gesture") +
  pub_theme +
  theme(legend.position = "none")
ggsave(here("figures", "fig_utterance_type_predicted.png"),
       fig_utt_predicted, width = 7, height = 5, dpi = 300)


# =============================================================================
# Q2. Does gesture probability increase with referential density?
#     Do nouns and pronouns interact?
# =============================================================================

# Main-effects model
m_rd_main <- glmer(
  Social_Cue ~ Noun_count + Pronoun_count + (1 | ID),
  data = data, family = binomial
)
cat("\n===== MAIN EFFECTS MODEL =====\n")
summary(m_rd_main)
Anova(m_rd_main, type = "III")

# Interaction model
m_rd_int <- glmer(
  Social_Cue ~ Noun_count * Pronoun_count + (1 | ID),
  data = data, family = binomial
)
cat("\n===== INTERACTION MODEL =====\n")
summary(m_rd_int)
Anova(m_rd_int, type = "III")

# Table: interaction model estimates with odds ratios
tab_rd <- tidy(m_rd_int, effects = "fixed", conf.int = TRUE, conf.level = 0.95) %>%
  mutate(
    Predictor = case_when(
      term == "(Intercept)"              ~ "Intercept (0 nouns, 0 pronouns)",
      term == "Noun_count"               ~ "Noun count (per token)",
      term == "Pronoun_count"            ~ "Pronoun count (per token)",
      term == "Noun_count:Pronoun_count" ~ "Noun x Pronoun interaction",
      TRUE                               ~ term
    ),
    OR      = exp(estimate),
    p.value = ifelse(p.value < .001, "< .001", sprintf("%.3f", p.value))
  ) %>%
  select(Predictor,
         Estimate       = estimate,
         SE             = std.error,
         OR,
         `95% CI Lower` = conf.low,
         `95% CI Upper` = conf.high,
         p              = p.value)
write_csv(tab_rd, here("tables", "Table_NounPron_Interaction.csv"))

# Figure: predicted probability of gesture across noun/pronoun counts
interaction_pred <- ggpredict(
  m_rd_int,
  terms = c("Noun_count [0:5]", "Pronoun_count [0:5]")
)

fig_rd_lines <- ggplot(interaction_pred,
                      aes(x = x, y = predicted, group = group,
                          colour = factor(group), fill = factor(group))) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1.3) +
  labs(x = "Noun Count",
       y = "Predicted Gesture Probability",
       colour = "Pronoun Count",
       fill   = "Pronoun Count") +
  pub_theme
ggsave(here("figures", "fig_referential_density_lines.png"),
       fig_rd_lines, width = 8, height = 5, dpi = 300)

fig_rd_heatmap <- ggplot(interaction_pred,
                         aes(x = x, y = group, fill = predicted)) +
  geom_tile() +
  scale_fill_gradient(name = "Predicted\nProbability", limits = c(0, 1)) +
  labs(x = "Noun Count", y = "Pronoun Count") +
  pub_theme
ggsave(here("figures", "fig_referential_density_heatmap.png"),
       fig_rd_heatmap, width = 7, height = 5, dpi = 300)


# ---- Sub-analyses: count effects within single-class utterances -------------

noun_only <- data %>% filter(Noun_count > 0 & Pronoun_count == 0)
m_noun_only <- glm(Social_Cue ~ Noun_count, data = noun_only, family = binomial)
cat("\n===== NOUN-ONLY UTTERANCES =====\n")
summary(m_noun_only)

pronoun_only <- data %>% filter(Pronoun_count > 0 & Noun_count == 0)
m_pron_only <- glm(Social_Cue ~ Pronoun_count, data = pronoun_only, family = binomial)
cat("\n===== PRONOUN-ONLY UTTERANCES =====\n")
summary(m_pron_only)


# =============================================================================
# Q3. Does syntactic complexity (MLU) predict gesture use?
#     Does the relation depend on utterance type?
# =============================================================================

mlu_data <- data %>% filter(!is.na(MLU))

# MLU main effect (MLU is z-scored; coefficients are per 1 SD).
m_mlu <- glmer(Social_Cue ~ MLU_z + (1 | ID),
               data = mlu_data, family = binomial)
summary(m_mlu)
Anova(m_mlu, type = "III")

mlu_pred <- ggpredict(m_mlu, terms = "MLU_z")
fig_mlu <- plot(mlu_pred) +
  labs(x = "Standardized MLU (z-score)",
       y = "Predicted Probability of Gesture") +
  pub_theme
ggsave(here("figures", "fig_mlu_main.png"),
       fig_mlu, width = 7, height = 5, dpi = 300)

# MLU x utterance_type interaction (MLU is z-scored).
m_mlu_int <- glmer(
  Social_Cue ~ MLU_z * utterance_type + (1 | ID),
  data = mlu_data, family = binomial
)
summary(m_mlu_int)
Anova(m_mlu_int, type = "III")

# Per-cell simple slopes of MLU_z (per 1 SD), with SE and p-value.
# These give the within-utterance-type MLU effects reported in the manuscript.
mlu_simple_slopes <- emtrends(m_mlu_int, ~ utterance_type, var = "MLU_z")
print(summary(mlu_simple_slopes, infer = c(TRUE, TRUE)))

mlu_effects <- ggpredict(m_mlu_int, terms = c("MLU_z", "utterance_type"))
fig_mlu_int <- ggplot(mlu_effects,
                      aes(x = x, y = predicted, colour = group, fill = group)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1.3) +
  labs(x = "Standardized MLU (z-score)",
       y = "Predicted Gesture Probability",
       colour = "Utterance Type",
       fill   = "Utterance Type") +
  pub_theme
ggsave(here("figures", "fig_mlu_by_utterance_type.png"),
       fig_mlu_int, width = 8, height = 5, dpi = 300)


# =============================================================================
# Q4. Within mixed (noun + pronoun) utterances, do same-referent pairs
#     attract more gestures than different-referent pairs?
# =============================================================================

ref_data <- data %>% filter(!is.na(Same_Referent) & !is.na(Social_Cue))

# Descriptive proportion by same/different referent
referent_summary <- ref_data %>%
  group_by(Same_Referent) %>%
  summarise(
    Proportion_Social_Cue = mean(Social_Cue),
    Count                 = n(),
    .groups               = "drop"
  )
print(referent_summary)

# Chi-square test
referent_table <- table(ref_data$Same_Referent, ref_data$Social_Cue)
print(referent_table)
print(chisq.test(referent_table))

# Mixed-effects model
m_ref <- glmer(Social_Cue ~ Same_Referent + (1 | ID),
               data = ref_data, family = binomial)
summary(m_ref)
Anova(m_ref, type = "III")

# Mixed-effects model controlling for noun/pronoun counts
m_ref_ctrl <- glmer(
  Social_Cue ~ Same_Referent + Noun_count + Pronoun_count + (1 | ID),
  data = ref_data, family = binomial
)
summary(m_ref_ctrl)

# Figure: descriptive proportion by same/different referent
fig_referent <- ggplot(referent_summary,
                       aes(x = factor(Same_Referent),
                           y = Proportion_Social_Cue,
                           fill = factor(Same_Referent))) +
  geom_col(alpha = 0.8) +
  geom_text(aes(label = round(Proportion_Social_Cue, 2)), vjust = -0.5) +
  labs(x = "Same Referent (1 = Yes, 0 = No)",
       y = "Proportion of Utterances with Gesture") +
  pub_theme +
  theme(legend.position = "none")
ggsave(here("figures", "fig_same_referent.png"),
       fig_referent, width = 6, height = 5, dpi = 300)


# ---- Session info -----------------------------------------------------------
sessionInfo()
