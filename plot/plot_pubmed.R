require("ggplot2")
require("data.table")


# Add sequence ID
add_sid <- function(dt) {
  dt[, freq2 := shift(freq, 1, type="lead", fill=0.5)]
  dt$diffSeries <- dt$freq > dt$freq2
  dt$sid <- cumsum(dt$diffSeries)
  dt$sid <- shift(dt$sid, 1, type="lag", fill=0)
  dt[, .(freq, power, sid)]
}


##
# Mistral as estimator
##
# GPT-4
d.gpt4.orig <- fread("../data/gpt-4/pubmed_gpt-4.original.mistral.fftnorm.txt")
d.gpt4.samp <- fread("../data/gpt-4/pubmed_gpt-4.sampled.mistral.fftnorm.txt")
d.gpt4.orig <- add_sid(d.gpt4.orig)
d.gpt4.samp <- add_sid(d.gpt4.samp)
d.gpt4.orig$type <- "Human"
d.gpt4.samp$type <- "GPT-4"
d.gpt4 <- rbind(d.gpt4.orig, d.gpt4.samp)
# Smoothed plot
p <- ggplot(d.gpt4, aes(freq, power, color=type)) +
    geom_smooth() +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("PubMed: Human vs. GPT-4") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("gpt4_human_pubmed_mistral_norm.pdf", plot=p, width=5, height=5)

# GPT3.5
d.gpt3.5.orig <- fread("../data/gpt-3.5/pubmed_gpt-3.5-turbo.original.mistral.fftnorm.txt")
d.gpt3.5.samp <- fread("../data/gpt-3.5/pubmed_gpt-3.5-turbo.sampled.mistral.fftnorm.txt")
d.gpt3.5.orig <- add_sid(d.gpt3.5.orig)
d.gpt3.5.samp <- add_sid(d.gpt3.5.samp)
d.gpt3.5.orig$type <- "Human"
d.gpt3.5.samp$type <- "GPT-3.5"
d.gpt3.5 <- rbind(d.gpt3.5.orig, d.gpt3.5.samp)
# Smoothed plot
p <- ggplot(d.gpt3.5, aes(freq, power, color=type)) +
    geom_smooth() +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("PubMed: Human vs. GPT-3.5") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("gpt3.5_human_pubmed_mistral_norm.pdf", plot=p, width=5, height=5)

# Davinci
d.davinci.orig <- fread("../data/davinci/pubmed_davinci.original.mistral.fftnorm.txt")
d.davinci.samp <- fread("../data/davinci/pubmed_davinci.sampled.mistral.fftnorm.txt")
d.davinci.orig <- add_sid(d.davinci.orig)
d.davinci.samp <- add_sid(d.davinci.samp)
d.davinci.orig$type <- "Human"
d.davinci.samp$type <- "Davinci"
d.davinci <- rbind(d.davinci.orig, d.davinci.samp)
# Smoothed plot
p <- ggplot(d.davinci, aes(freq, power, color=type)) +
    geom_smooth() +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("PubMed: Human vs. Davinci") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("davinci_human_pubmed_mistral_norm.pdf", plot=p, width=5, height=5)


##
# GPT2xl as estimator
##
# GPT-4
d.gpt4.orig <- fread("../data/gpt-4/pubmed_gpt-4.original.gpt2xl.fft.txt")
d.gpt4.samp <- fread("../data/gpt-4/pubmed_gpt-4.sampled.gpt2xl.fft.txt")
d.gpt4.orig <- add_sid(d.gpt4.orig)
d.gpt4.samp <- add_sid(d.gpt4.samp)
d.gpt4.orig$type <- "Human"
d.gpt4.samp$type <- "GPT-4"
d.gpt4 <- rbind(d.gpt4.orig, d.gpt4.samp)
# Smoothed plot
p <- ggplot(d.gpt4, aes(freq, power, color=type)) +
    geom_smooth() +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("PubMed: Human vs. GPT-4 \n(estimated w/ GPT2xl)") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("gpt4_human_pubmed_gpt2xl.pdf", plot=p, width=5, height=5)



###
# Distributions of spectral power, especially over the lower 15% of spectrum
###

# GPT-4
lower_freq_ratio <- 0.15
pow.gpt4 <- d.gpt4[freq <= lower_freq_ratio * 0.5, .(power, sid, type)]
p <- ggplot(pow.gpt4, aes(log(abs(power) + 1), fill=type)) +
    geom_density(alpha=0.5) +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("PubMed: Human vs. GPT-4 \n(estimated w/ Mistral)") +
    labs(x = "Power", y = "Density")
ggsave("gpt4_human_pubmed_mistral_norm_density(lower0.15).pdf", plot=p, width=5, height=5)