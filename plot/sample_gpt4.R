require("ggplot2")
require("data.table")

# Unnormalized data
d.human <- fread("../data/demo_human.fft.txt")
d.gpt4 <- fread("../data/gpt-4/writing_gpt-4.gpt2.fft.txt")
d.gpt3.5 <- fread("../data/gpt-3.5/writing_gpt-3.5-turbo.gpt2.fft.txt")
d.davinci <- fread("../data/davinci/writing_davinci.gpt2.fft.txt")

# Normalized data
d.human <- fread("../data/demo_human.fftnorm.txt")
d.gpt4 <- fread("../data/gpt-4/writing_gpt-4.gpt2.fftnorm.txt")
d.gpt3.5 <- fread("../data/gpt-3.5/writing_gpt-3.5-turbo.gpt2.fftnorm.txt")
d.davinci <- fread("../data/davinci/writing_davinci.gpt2.fftnorm.txt")


# Add sequence ID
add_sid <- function(dt) {
  dt[, freq2 := shift(freq, 1, type="lead", fill=0.5)]
  dt$diffSeries <- dt$freq > dt$freq2
  dt$sid <- cumsum(dt$diffSeries)
  dt$sid <- shift(dt$sid, 1, type="lag", fill=0)
  dt
}

d.human <- add_sid(d.human)
d.gpt4 <- add_sid(d.gpt4)
d.gpt3.5 <- add_sid(d.gpt3.5)
d.davinci <- add_sid(d.davinci)

# Add type column
d.human$type <- "human"
d.gpt4$type <- "gpt-4"
d.gpt3.5$type <- "gpt-3.5"
d.davinci$type <- "davinci"
d.combined <- rbind(d.human, d.gpt4, d.gpt3.5, d.davinci)

# Smoothed plot
p <- ggplot(d.combined, aes(freq, power, color=type)) +
    geom_smooth() +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. Model") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("gpt4_human_writing_spectrum_smoothed_norm.pdf", plot=p, width=5, height=5)

# Plot gpt-4, gpt-3.5, davinci
p <- ggplot(d.combined[type!="human"], aes(freq, power, color=type)) +
    geom_smooth() + 
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("GPT-4 vs. GPT-3.5 vs. Davinci") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("gpt4_gpt3.5_davinci_writing_spectrum_smoothed_norm.pdf", plot=p, width=5, height=5)


# Plot gpt-4, gpt-3.5, davinci with a poor model (gpt2)
d.gpt2 <- fread("../data/demo_model.fftnorm.txt")
d.gpt2 <- add_sid(d.gpt2)
d.gpt2$type <- "gpt-2"
d.combined2 <- rbind(d.gpt4, d.gpt3.5, d.davinci, d.gpt2)

p <- ggplot(d.combined2, aes(freq, power, color=type)) +
    geom_smooth() + 
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("GPT-4 vs. GPT-3.5 vs. Davinci vs. GPT-2") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("gpt4_gpt3.5_davinci_gpt2_writing_spectrum_smoothed_norm.pdf", plot=p, width=5, height=5)


# Plot gpt-2 and human together
d.combined3 <- rbind(d.human, d.gpt2)
p <- ggplot(d.combined3, aes(freq, power, color=type)) +
    geom_smooth() + 
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. GPT-2") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("human_gpt2_writing_spectrum_smoothed_norm.pdf", plot=p, width=5, height=5)