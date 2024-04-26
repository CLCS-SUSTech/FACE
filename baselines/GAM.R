require("ggplot2")
require("data.table")
require("mgcv") # For fitting GAM
require("lsa") # For computing cosine similarity


# Add sequence ID
add_sid <- function(dt) {
    dt[, freq2 := shift(freq, 1, type="lead", fill=0.5)]
    dt$diffSeries <- dt$freq > dt$freq2
    dt$sid <- cumsum(dt$diffSeries)
    dt$sid <- shift(dt$sid, 1, type="lag", fill=0)
    dt
}


# The function that returns predictions from GAM model
get_GAM_pred <- function(gam_model) {
    x <- data.table(freq = seq(0, 0.5, 0.001))
    y <- as.numeric(predict(gam_model, x))
    d <- data.table(x = x$freq, y = y)
    return(d)
}

# Read data
# GPT-4
d.gpt4.orig <- fread("writing_unigram_mincount=3/writing_gpt-4.original.unigram.fftnorm.txt")
d.gpt4.samp <- fread("writing_unigram_mincount=3/writing_gpt-4.sampled.unigram.fftnorm.txt")
d.gpt4.orig$type <- "Human"
d.gpt4.samp$type <- "GPT-4"

gam.gpt4.orig <- gam(power ~ s(freq, bs="cs"), data = d.gpt4.orig)
pred.gpt4.orig <- get_GAM_pred(gam.gpt4.orig)
pred.gpt4.orig$type <- "Human"

gam.gpt4.samp <- gam(power ~ s(freq, bs="cs"), data = d.gpt4.samp)
pred.gpt4.samp <- get_GAM_pred(gam.gpt4.samp)
pred.gpt4.samp$type <- "GPT-4"

pred.gpt4 <- rbind(pred.gpt4.orig, pred.gpt4.samp)

p <- ggplot(pred.gpt4, aes(x, y)) +
    geom_line(aes(color=type, linetype=type)) +
    ggtitle("Human vs. GPT-4") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k]))) + 
    theme_bw()
ggsave("gam_gpt4_human_writing_fftnorm.pdf", plot=p, width=5, height=5)


# GPT-3.5
d.gpt3.5.orig <- fread("writing_unigram_mincount=3/writing_gpt-3.5-turbo.original.unigram.fftnorm.txt")
d.gpt3.5.samp <- fread("writing_unigram_mincount=3/writing_gpt-3.5-turbo.sampled.unigram.fftnorm.txt")
d.gpt3.5.orig$type <- "Human"
d.gpt3.5.samp$type <- "GPT-3.5"

gam.gpt3.5.orig <- gam(power ~ s(freq, bs="cs"), data = d.gpt3.5.orig)
pred.gpt3.5.orig <- get_GAM_pred(gam.gpt3.5.orig)
pred.gpt3.5.orig$type <- "Human"

gam.gpt3.5.samp <- gam(power ~ s(freq, bs="cs"), data = d.gpt3.5.samp)
pred.gpt3.5.samp <- get_GAM_pred(gam.gpt3.5.samp)
pred.gpt3.5.samp$type <- "GPT-3.5"

pred.gpt3.5 <- rbind(pred.gpt3.5.orig, pred.gpt3.5.samp)

p <- ggplot(pred.gpt3.5, aes(x, y)) +
    geom_line(aes(color=type, linetype=type)) +
    ggtitle("Human vs. GPT-3.5") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))+ 
    theme_bw()
ggsave("gam_gpt3.5_human_writing_fftnorm.pdf", plot=p, width=5, height=5)


# Davinci
d.davinci.orig <- fread("writing_unigram_mincount=3/writing_davinci.original.unigram.fftnorm.txt")
d.davinci.samp <- fread("writing_unigram_mincount=3/writing_davinci.sampled.unigram.fftnorm.txt")
d.davinci.orig$type <- "Human"
d.davinci.samp$type <- "Davinci"

gam.davinci.orig <- gam(power ~ s(freq, bs="cs"), data = d.davinci.orig)
pred.davinci.orig <- get_GAM_pred(gam.davinci.orig)
pred.davinci.orig$type <- "Human"

gam.davinci.samp <- gam(power ~ s(freq, bs="cs"), data = d.davinci.samp)
pred.davinci.samp <- get_GAM_pred(gam.davinci.samp)
pred.davinci.samp$type <- "Davinci"

pred.davinci <- rbind(pred.davinci.orig, pred.davinci.samp)

p <- ggplot(pred.davinci, aes(x, y)) +
    geom_line(aes(color=type, linetype=type)) +
    ggtitle("Human vs. Davinci") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))+ 
    theme_bw()
ggsave("gam_davinci_human_writing_fftnorm.pdf", plot=p, width=5, height=5)


# Plot GPT-4, GPT-3.5, Davinci, and Human together
pred.all <- rbind(pred.gpt4[type == "GPT-4"],
                  pred.gpt3.5[type == "GPT-3.5"],
                  pred.davinci[type == "Davinci"],
                  data.table(x = pred.gpt4.orig$x, 
                             y = pred.gpt4.orig$y + pred.gpt3.5.orig$y + pred.davinci.orig$y, 
                             type = "Human"))

p <- ggplot(pred.all, aes(x, y)) +
    geom_line(aes(color=type, linetype=type)) +
    ggtitle("Human vs. GPT-4 vs. GPT-3.5 vs. Davinci") +
    labs(x = bquote(omega[k]), y = bquote(X(omega[k])))+ 
    theme_bw()
ggsave("gam_all_human_writing_fftnorm.pdf", plot=p, width=5, height=5)



# Measure distance between predicted y (power) and ground truth
# human => 0, model => 1
d.gpt4.orig <- add_sid(d.gpt4.orig)
count.00 <- 0
count.01 <- 0
for (i in unique(d.gpt4.orig$sid)) {
    d.tmp <- d.gpt4.orig[sid == i]
    y <- d.gpt4.orig[sid == i]$power
    y_pred_orig <- predict.gam(gam.gpt4.orig, d.tmp)
    y_pred_orig <- as.numeric(y_pred_orig)
    y_pred_orig <- y_pred_orig * sd(gam.gpt4.orig$y) + mean(gam.gpt4.orig$y)
    y_pred_samp <- predict.gam(gam.gpt4.samp, d.tmp)
    y_pred_samp <- as.numeric(y_pred_samp)
    y_pred_samp <- y_pred_samp * sd(gam.gpt4.samp$y) + mean(gam.gpt4.samp$y)
    if (sum(abs(y[1:5] - y_pred_orig[1:5])) < sum(abs(y[1:5] - y_pred_samp[1:5]))) {
        count.00 <- count.00 + 1
    } else {
        count.01 <- count.01 + 1
    }
}

d.gpt4.samp <- add_sid(d.gpt4.samp)
count.10 <- 0
count.11 <- 0
for (i in unique(d.gpt4.samp$sid)) {
    d.tmp <- d.gpt4.samp[sid == i]
    y <- d.gpt4.samp[sid == i]$power
    y_pred_samp <- predict.gam(gam.gpt4.samp, d.tmp)
    y_pred_samp <- as.numeric(y_pred_samp)
    y_pred_samp <- y_pred_samp * sd(gam.gpt4.samp$y) + mean(gam.gpt4.samp$y)
    y_pred_orig <- predict.gam(gam.gpt4.orig, d.tmp)
    y_pred_orig <- as.numeric(y_pred_orig)
    y_pred_orig <- y_pred_orig * sd(gam.gpt4.orig$y) + mean(gam.gpt4.orig$y)
    if (sum(abs(y[1:5] - y_pred_orig[1:5])) < sum(abs(y[1:5] - y_pred_samp[1:5]))) {
        count.10 <- count.10 + 1
    } else {
        count.11 <- count.11 + 1
    }
}
acc <- (count.00 + count.11) / (count.00 + count.01 + count.10 + count.11)
# 以上实验证明，简单的欧式距离是不够的，需要 overlap based, 比如 FACE-SO  