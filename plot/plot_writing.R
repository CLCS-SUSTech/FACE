require("ggplot2")
require("data.table")


# Add sequence ID
add_sid <- function(dt) {
  dt[, freq2 := shift(freq, 1, type="lead", fill=0.5)]
  dt$diffSeries <- dt$freq > dt$freq2
  dt$sid <- cumsum(dt$diffSeries)
  dt$sid <- shift(dt$sid, 1, type="lag", fill=0)
  dt
}

##
# GPT2xl as estimator
##

# GPT-4
d.gpt4.orig <- fread("../data/gpt-4/writing_gpt-4.original.gpt2xl.fftnorm.txt")
d.gpt4.samp <- fread("../data/gpt-4/writing_gpt-4.sampled.gpt2xl.fftnorm.txt")
d.gpt4.orig <- add_sid(d.gpt4.orig)
d.gpt4.samp <- add_sid(d.gpt4.samp)
d.gpt4.orig$type <- "Human"
d.gpt4.samp$type <- "GPT-4"
d.gpt4 <- rbind(d.gpt4.orig, d.gpt4.samp)
# Smoothed plot
p <- ggplot(d.gpt4, aes(freq, power, color=type)) +
    geom_smooth() +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. GPT-4") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("gpt4_human_writing_gpt2xl_norm.pdf", plot=p, width=5, height=5)

# GPT3.5
d.gpt3.5.orig <- fread("../data/gpt-3.5/writing_gpt-3.5-turbo.original.gpt2xl.fftnorm.txt")
d.gpt3.5.samp <- fread("../data/gpt-3.5/writing_gpt-3.5-turbo.sampled.gpt2xl.fftnorm.txt")
d.gpt3.5.orig <- add_sid(d.gpt3.5.orig)
d.gpt3.5.samp <- add_sid(d.gpt3.5.samp)
d.gpt3.5.orig$type <- "Human"
d.gpt3.5.samp$type <- "GPT-3.5"
d.gpt3.5 <- rbind(d.gpt3.5.orig, d.gpt3.5.samp)
# Smoothed plot
p <- ggplot(d.gpt3.5, aes(freq, power, color=type)) +
    geom_smooth() +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. GPT-3.5") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("gpt3.5_human_writing_gpt2xl_norm.pdf", plot=p, width=5, height=5)

# Davinci
d.davinci.orig <- fread("../data/davinci/writing_davinci.original.gpt2xl.fftnorm.txt")
d.davinci.samp <- fread("../data/davinci/writing_davinci.sampled.gpt2xl.fftnorm.txt")
d.davinci.orig <- add_sid(d.davinci.orig)
d.davinci.samp <- add_sid(d.davinci.samp)
d.davinci.orig$type <- "Human"
d.davinci.samp$type <- "Davinci"
d.davinci <- rbind(d.davinci.orig, d.davinci.samp)
# Smoothed plot
p <- ggplot(d.davinci, aes(freq, power, color=type)) +
    geom_smooth() +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. Davinci") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("davinci_human_writing_gpt2xl_norm.pdf", plot=p, width=5, height=5)

# Plot GPT-4, GPT-3.5, Davinci, and Human together
d.all <- rbind(d.gpt4, d.gpt3.5, d.davinci.samp)
p <- ggplot(d.all, aes(freq, power, color=type)) +
    geom_smooth() +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. GPT-4 vs. GPT-3.5 vs. Davinci") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("all_human_writing_gpt2xl_norm.pdf", plot=p, width=5, height=5)

# Plot GPT-4, GPT-3.5, and Human (from GPT-4 and 3.5) together
d.gpts <- rbind(d.gpt4, d.gpt3.5)
p <- ggplot(d.gpts, aes(freq, power, color=type)) +
    geom_smooth() +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. GPT-4 vs. GPT-3.5") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("gpts_human_writing_gpt2xl_norm.pdf", plot=p, width=5, height=5)

# GPT2 from demo
d.gpt2.orig <- fread("../data/demo_human.gpt2xl.fftnorm.txt")
d.gpt2.samp <- fread("../data/demo_model.gpt2xl.fftnorm.txt")


##
# GPT2 (small) as estimator
##


##
# Unigram baseline
##
# GPT-4
d.gpt4.unigram.orig <- fread("../baselines/writing_gpt-4.original.unigram.fft.txt")
d.gpt4.unigram.samp <- fread("../baselines/writing_gpt-4.sampled.unigram.fft.txt")
d.gpt4.unigram.orig <- add_sid(d.gpt4.unigram.orig)
d.gpt4.unigram.samp <- add_sid(d.gpt4.unigram.samp)
d.gpt4.unigram.orig$type <- "Human"
d.gpt4.unigram.samp$type <- "GPT-4"
d.gpt4.unigram <- rbind(d.gpt4.unigram.orig, d.gpt4.unigram.samp)
# Smoothed plot
p <- ggplot(d.gpt4.unigram, aes(freq, power, color=type)) +
    geom_smooth() +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. GPT-4") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("gpt4_human_writing_unigram.pdf", plot=p, width=5, height=5)

# GPT3.5
d.gpt3.5.unigram.orig <- fread("../baselines/writing_gpt-3.5-turbo.original.unigram.fft.txt")
d.gpt3.5.unigram.samp <- fread("../baselines/writing_gpt-3.5-turbo.sampled.unigram.fft.txt")
d.gpt3.5.unigram.orig <- add_sid(d.gpt3.5.unigram.orig)
d.gpt3.5.unigram.samp <- add_sid(d.gpt3.5.unigram.samp)
d.gpt3.5.unigram.orig$type <- "Human"
d.gpt3.5.unigram.samp$type <- "GPT-3.5"
d.gpt3.5.unigram <- rbind(d.gpt3.5.unigram.orig, d.gpt3.5.unigram.samp)
# Smoothed plot
p <- ggplot(d.gpt3.5.unigram, aes(freq, power, color=type)) +
    geom_smooth() +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. GPT-3.5") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("gpt3.5_human_writing_unigram.pdf", plot=p, width=5, height=5)

# Davinci
d.davinci.unigram.orig <- fread("../baselines/writing_davinci.original.unigram.fft.txt")
d.davinci.unigram.samp <- fread("../baselines/writing_davinci.sampled.unigram.fft.txt")
d.davinci.unigram.orig <- add_sid(d.davinci.unigram.orig)
d.davinci.unigram.samp <- add_sid(d.davinci.unigram.samp)
d.davinci.unigram.orig$type <- "Human"
d.davinci.unigram.samp$type <- "Davinci"
d.davinci.unigram <- rbind(d.davinci.unigram.orig, d.davinci.unigram.samp)
# Smoothed plot
p <- ggplot(d.davinci.unigram, aes(freq, power, color=type)) +
    geom_smooth() +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. Davinci") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("davinci_human_writing_unigram.pdf", plot=p, width=5, height=5)

# Plot GPT-4, GPT-3.5, Davinci, and Human together
d.all.unigram <- rbind(d.gpt4.unigram, d.gpt3.5.unigram, d.davinci.unigram)
p <- ggplot(d.all.unigram, aes(freq, power, color=type)) +
    geom_smooth() +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. GPT-4 vs. GPT-3.5 vs. Davinci") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("all_human_writing_unigram.pdf", plot=p, width=5, height=5)