require("data.table")
require("stringr")
require("readr")

# Add sequence ID
add_sid <- function(dt) {
    dt[, freq2 := shift(freq, 1, type="lead", fill=0.5)]
    dt$diffSeries <- dt$freq > dt$freq2
    dt$sid <- cumsum(dt$diffSeries)
    dt$sid <- shift(dt$sid, 1, type="lag", fill=0)
    dt
}

# The function for reading nll.txt files
read_nll_like_file <- function(file_path) {
    read_lines(file_path) %>%
        str_split(" ") %>%
        unlist() %>%
        as.numeric() %>% 
        data.table(value = .)
}
# Read and normalize
read_nll <- function(file_path, norm=FALSE) {
    nll_list <- c()
    len_list <- c()
    conn <- file(file_path, "r")
    sid <- 0
    while (TRUE) {
        line <- readLines(conn, n=1)
        if (length(line) == 0) {
            break
        }
        nll <- as.numeric(strsplit(line, " ")[[1]])
        if (norm) {
            nll <- (nll - mean(nll)) / sd(nll)
        }
        nll_list <- c(nll_list, nll)
        len_list <- c(len_list, rep(sid, length(nll)))
        sid <- sid + 1
    }
    close(conn)
    data.table(value = as.numeric(nll_list), sid = as.integer(len_list))
}


# Analyze NLL distributions
# min_count=1,3
nll.gpt4.orig <- read_nll_like_file("writing_unigram_mincount=3/writing_gpt-4.original.unigram.nll.txt")
nll.gpt4.samp <- read_nll_like_file("writing_unigram_mincount=3/writing_gpt-4.sampled.unigram.nll.txt")
nll.gpt4.orig$type <- "Human"
nll.gpt4.samp$type <- "GPT-4"
nll.gpt4 <- rbind(nll.gpt4.orig, nll.gpt4.samp)
# Plot
p <- ggplot(nll.gpt4, aes(value, fill=type)) +
    geom_density(alpha=0.5) +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. GPT-4") +
    labs(x = "NLL", y = "Density")
ggsave("gpt4_human_writing_nlldistr_unigram_mincount=3.pdf", plot=p, width=5, height=5)

# nll normalized within same sentence
nllnorm.gpt4.orig <- read_nll("writing_unigram_mincount=3/writing_gpt-4.original.unigram.nll.txt", norm=TRUE)
nllnorm.gpt4.samp <- read_nll("writing_unigram_mincount=3/writing_gpt-4.sampled.unigram.nll.txt", norm=TRUE)
nllnorm.gpt4.orig$type <- "Human"
nllnorm.gpt4.samp$type <- "GPT-4"
nllnorm.gpt4 <- rbind(nllnorm.gpt4.orig, nllnorm.gpt4.samp)
# Plot
p <- ggplot(nllnorm.gpt4, aes(value, fill=type)) +
    geom_density(alpha=0.5) +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. GPT-4") +
    labs(x = "NLL (normalized)", y = "Density")
ggsave("gpt4_human_writing_nllnormdistr_unigram_mincount=3.pdf", plot=p, width=5, height=5)


# Distribution of FFT power
# min_count=3
fftnorm.gpt4.orig <- fread("writing_unigram_mincount=3/writing_gpt-4.original.unigram.fftnorm.txt")
fftnorm.gpt4.samp <- fread("writing_unigram_mincount=3/writing_gpt-4.sampled.unigram.fftnorm.txt")
fftnorm.gpt4.orig$type <- "Human"
fftnorm.gpt4.samp$type <- "GPT-4"
fftnorm.gpt4 <- rbind(fftnorm.gpt4.orig, fftnorm.gpt4.samp)
# Plot
p <- ggplot(fftnorm.gpt4, aes(power, fill=type)) +
    geom_density(alpha=0.5) +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. GPT-4") +
    labs(x = "FFT Normalzied Power", y = "Density")
ggsave("gpt4_human_writing_fftnormdistr_unigram_mincount=3.pdf", plot=p, width=5, height=5)


# Simple t-test grouped by frequency
t.test(fftnorm.gpt4.orig[,power], fftnorm.gpt4.samp[,power]) # Not distinguishable

t.test(fftnorm.gpt4.orig[freq==0,power], fftnorm.gpt4.samp[freq==0,power]) # Not distinguishable
# t = 1.2774e-14, df = 296.16, p-value = 1

t.test(fftnorm.gpt4.orig[between(freq, 0, 0.01),power],
       fftnorm.gpt4.samp[between(freq, 0, 0.01),power]) # Distinguishable
# t = -2.1774, df = 597.3, p-value = 0.02984
# gpt4.orig < gpt4.samp in terms of power near 0

t.test(fftnorm.gpt4.orig[between(freq, 0.01, 0.02),power], 
       fftnorm.gpt4.samp[between(freq, 0.01, 0.02),power]) #ND 

t.test(fftnorm.gpt4.orig[between(freq, 0, 0.02),power], 
       fftnorm.gpt4.samp[between(freq, 0, 0.02),power]) # Not distinguishable
t.test(fftnorm.gpt4.orig[between(freq, 0.06, 0.075),power], 
       fftnorm.gpt4.samp[between(freq, 0.06, 0.075),power]) # Not distinguishable

# Plot FFT normalized power at freq between 0 and 0.01
p <- ggplot(fftnorm.gpt4[between(freq, 0, 0.01),], aes(power, fill=type)) +
    geom_density(alpha=0.5) +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. GPT-4 (0-0.01)") +
    labs(x = "FFT Normalzied Power", y = "Density")
ggsave("gpt4_human_writing_fftnormdistr_0-0.01_unigram_mincount=3.pdf", plot=p, width=5, height=5)


# Try fitting linear models
# No, linear model is not working for unigram data
lm.gpt4.orig <- lm(power ~ freq, data = fftnorm.gpt4.orig[freq>0])
lm.gpt4.samp <- lm(power ~ freq, data = fftnorm.gpt4.samp[freq>0])
summary(lm.gpt4.orig)

# Plot the fitted lines with points
p <- ggplot(fftnorm.gpt4, aes(freq, power, color=type)) +
    geom_point() +
    geom_abline(intercept=coef(lm.gpt4.orig)[1], slope=coef(lm.gpt4.orig)[2], linetype="dashed") +
    geom_abline(intercept=coef(lm.gpt4.samp)[1], slope=coef(lm.gpt4.samp)[2], linetype="dashed") +
    theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
    ggtitle("Human vs. GPT-4") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("gpt4_human_writing_fftnorm_lm_unigram_mincount=3.pdf", plot=p, width=5, height=5)

gpt4.orig.intercept <- coef(lm.gpt4.orig)[1]
gpt4.samp.intercept <- coef(lm.gpt4.samp)[1]
