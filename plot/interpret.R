require("ggplot2")
require("data.table")
require("mgcv")
require("splus2R") # For the peaks() function
require("patchwork")

####
# Find typical spectrum from all 3 models: gpt2-xl, opt, bloom, and gold standard
d.gpt2.fft <- fread("../data/data_gpt2_old/xl-1542M.test.model=gpt2.fft.csv")
d.opt.fft <- fread("../data/experiments_data/opt/webtext.train_opt_6.7b_top_50_news.sorted.split.400.fft.csv")
d.bloom.fft <- fread("../data/experiments_data/bloom/split_news/webtext.train.model=.bloom_7b1.news.sorted.split.400.fft.csv")
d.gs.fft <- fread("../data/gs_james/gs_wiki/webtext.train.model=.wiki_3.fft.csv")


# Add sid to each series
add_sid <- function(dt) {
  dt[, freq2 := shift(freq, 1, type="lead", fill=0.5)]
  dt$diffSeries <- dt$freq > dt$freq2
  dt$sid <- cumsum(dt$diffSeries)
  dt$sid <- shift(dt$sid, 1, type="lag", fill=0)
  dt
}
# Calculate number of series
d.gpt2.fft <- add_sid(d.gpt2.fft)
length(unique(d.gpt2.fft$sid)) # 5000

d.gs.fft <- add_sid(d.gs.fft)
length(unique(d.gs.fft$sid)) # 5000


# Four colors
# gpt2-xl: #1f77b4
# opt: #ff7f0e
# bloom: #2ca02c
# gs: #d62728
colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")

p.gpt2 <- ggplot(d.gpt2.fft, aes(freq, power)) +
  geom_smooth(color="grey28") +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 20)) +
  ggtitle("GPT2-xl") +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("typical_spectrum_gpt2-xl.pdf", plot=p.gpt2, width=4, height=4)

p.opt <- ggplot(d.opt.fft, aes(freq, power)) +
  geom_smooth(color="grey28") +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 20)) +
  ggtitle("OPT") +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("typical_spectrum_opt.pdf", plot=p.opt, width=4, height=4)

p.bloom <- ggplot(d.bloom.fft, aes(freq, power)) +
  geom_smooth(color="grey28") +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 20)) +
  ggtitle("BLOOM") +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("typical_spectrum_bloom.pdf", plot=p.bloom, width=4, height=4)

p.gs <- ggplot(d.gs.fft, aes(freq, power)) +
  geom_smooth(color=colors[3]) +
  ggtitle("Human") +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("typical_spectrum_gs.pdf", plot=p.gs, width=4, height=4)

p <- p.gpt2 + p.opt + p.bloom + p.gs + plot_layout(ncol=2)
ggsave("typical_spectra.pdf", plot=p, width=8, height=8)


# Find the peaks on human spectrum
# Fit the GAM model
d.gs.fft.gam <- gam(power ~ s(freq, bs="cs"), data=d.gs.fft)
# Get predicted values on testing data
test <- data.frame(freq = seq(0, 0.5, 0.001))
test$power <- predict(d.gs.fft.gam, test)
test$freq[peaks(test$power)]
test$power[peaks(test$power)]

# annotate the peaks and pits on p.gs
peak_xs <- test$freq[peaks(test$power)]
peak_ys <- test$power[peaks(test$power)]
pit_xs <- test$freq[peaks(-test$power)]
pit_ys <- test$power[peaks(-test$power)]
peak_xs
# 0.12 0.23 0.34 0.45
pit_xs
# 0.06 0.18 0.29 0.40

p.gs.anno <- p.gs +
  geom_point(data=data.frame(x=peak_xs[1:2], y=peak_ys[1:2]), aes(x, y), color="red", size=3) +
  annotate("text", x=peak_xs[1]+0.07, y=peak_ys[1]+20,
           label=expression(omega[2]==0.12), parse=TRUE, color="red", size=5) +
  annotate("text", x=peak_xs[2]+0.05, y=peak_ys[2]+20,
           label=expression(omega[4]==0.23), parse=TRUE, color="red", size=5) +
  geom_point(data=data.frame(x=pit_xs[1:2], y=pit_ys[1:2]), aes(x, y), color="blue", shape=8, size=3) +
  annotate("text", x=pit_xs[1]+0.08, y=pit_ys[1]-5,
           label=expression(omega[1]==0.06), parse=TRUE, color="blue", size=5) +
  annotate("text", x=pit_xs[2]+0.08, y=pit_ys[2]-5,
           label=expression(omega[3]==0.18), parse=TRUE, color="blue", size=5)
ggsave("typical_spectrum_gs_anno.pdf", plot=p.gs.anno, width=4, height=4)


# fit gam model on gpt2-xl, opt, bloom, and compute the peaks and pits
d.gpt2.fft.gam <- gam(power ~ s(freq, bs="cs"), data=d.gpt2.fft)
d.opt.fft.gam <- gam(power ~ s(freq, bs="cs"), data=d.opt.fft)
d.bloom.fft.gam <- gam(power ~ s(freq, bs="cs"), data=d.bloom.fft)

test_gpt2 <- data.frame(freq = seq(0, 0.5, 0.001))
test_gpt2$power <- predict(d.gpt2.fft.gam, test_gpt2)
peak_xs_gpt2 <- test_gpt2$freq[peaks(test_gpt2$power)]
peak_ys_gpt2 <- test_gpt2$power[peaks(test_gpt2$power)]
pit_xs_gpt2 <- test_gpt2$freq[peaks(-test_gpt2$power)]
pit_ys_gpt2 <- test_gpt2$power[peaks(-test_gpt2$power)]

test_opt <- data.frame(freq = seq(0, 0.5, 0.001))
test_opt$power <- predict(d.opt.fft.gam, test_opt)
peak_xs_opt <- test_opt$freq[peaks(test_opt$power)]
peak_ys_opt <- test_opt$power[peaks(test_opt$power)]
pit_xs_opt <- test_opt$freq[peaks(-test_opt$power)]
pit_ys_opt <- test_opt$power[peaks(-test_opt$power)]

test_bloom <- data.frame(freq = seq(0, 0.5, 0.001))
test_bloom$power <- predict(d.bloom.fft.gam, test_bloom)
peak_xs_bloom <- test_bloom$freq[peaks(test_bloom$power)]
peak_ys_bloom <- test_bloom$power[peaks(test_bloom$power)]
pit_xs_bloom <- test_bloom$freq[peaks(-test_bloom$power)]
pit_ys_bloom <- test_bloom$power[peaks(-test_bloom$power)]


# add peaks and pits dots to p.gpt2, p.opt, p.bloom
p.gpt2.anno <- p.gpt2 +
  geom_point(data=data.frame(x=peak_xs_gpt2[1:2], y=peak_ys_gpt2[1:2]), aes(x, y),
             color="red", size=3) +
  geom_point(data=data.frame(x=pit_xs_gpt2[1:2], y=pit_ys_gpt2[1:2]), aes(x, y),
             color="blue", size=3, shape=8)
ggsave("typical_spectrum_GPT2xl_anno.pdf", plot=p.gpt2.anno, width=3, height=3)

p.opt.anno <- p.opt +
    geom_point(data=data.frame(x=peak_xs_opt[1:2], y=peak_ys_opt[1:2]), aes(x, y), color="red", size=3) +
    geom_point(data=data.frame(x=pit_xs_opt[1:2], y=pit_ys_opt[1:2]), aes(x, y),
               color="blue", size=3, shape=8)
ggsave("typical_spectrum_OPT_anno.pdf", plot=p.opt.anno, width=3, height=3)

p.bloom.anno <- p.bloom +
    geom_point(data=data.frame(x=peak_xs_bloom[1:2], y=peak_ys_bloom[1:2]), aes(x, y), color="red", size=3) +
    geom_point(data=data.frame(x=pit_xs_bloom[1:2], y=pit_ys_bloom[1:2]), aes(x, y),
               color="blue", size=3, shape=8)
ggsave("typical_spectrum_BLOOM_anno.pdf", plot=p.bloom.anno, width=3, height=3)

# Plot all annotated spectra together
p.anno <- p.gpt2.anno + p.opt.anno + p.bloom.anno + p.gs.anno + plot_layout(ncol=2)
ggsave("typical_spectra_anno.pdf", plot=p.anno, width=8, height=8)

p.anno_2 <- p.gpt2.anno + p.gs.anno + plot_layout(ncol=2)
ggsave("typical_spectra_anno_2.pdf", plot=p.anno_2, width=8, height=4)


###
# Pick one sid randomly and plot
###
# Read three data
d.gpt2xl <- fread("../data/data_gpt2_old/xl-1542M.test.model=gpt2.fft.csv")
d.gpt2sm <- fread("../data/data_gpt2_old/small-117M.test.model=gpt2.fft.csv")
# d.gs <- fread("../data/gs_james/gs_wiki/webtext.train.model=.wiki_3.fft.csv")
d.gs <- fread("../data/data_gpt2_old/webtext.test.model=gpt2.fft.csv")

# add sid
d.gpt2xl <- add_sid(d.gpt2xl)
d.gpt2sm <- add_sid(d.gpt2sm)
d.gs <- add_sid(d.gs)

set.seed(0)
d.gpt2xl.sample <- d.gpt2xl[sid == sample(sid, 1)]
d.gpt2sm.sample <- d.gpt2sm[sid == sample(sid, 1)]
d.gs.sample <- d.gs[sid == sample(sid, 1)]

# combine
d.gpt2xl.sample$source <- "GPT2-xl"
d.gpt2sm.sample$source <- "GPT2-sm"
d.gs.sample$source <- "human"
d.sample <- rbind(d.gpt2xl.sample, d.gpt2sm.sample, d.gs.sample)
d.sample$source <- factor(d.sample$source, levels=c("GPT2-xl", "GPT2-sm", "human"))

p.sample <- ggplot(d.sample[freq>0], aes(x=freq, y=power)) +
    geom_line(aes(color=source, linetype=source)) +
  theme_bw() + theme(legend.position="none") + ggtitle("Sample spectra") +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k]))) +
  scale_color_brewer(palette="Set1") +
  scale_linetype_manual(values=c("dashed", "dotted", "solid")) +
  facet_wrap(~source, ncol=1)
ggsave("sample_spectra.pdf", plot=p.sample, width=4, height=4)


# Fit GAM on d.gpt2xl, d.gpt2sm, d.gs
gam.gpt2xl <- gam(power ~ s(freq, bs="cs"), data=d.gpt2xl)
gam.gpt2sm <- gam(power ~ s(freq, bs="cs"), data=d.gpt2sm)
gam.gs <- gam(power ~ s(freq, bs="cs"), data=d.gs)

# get predicted values
x <- data.table(freq = seq(0, 0.5, 0.001))
y.gpt2xl <- as.numeric(predict(gam.gpt2xl, x))
y.gpt2sm <- as.numeric(predict(gam.gpt2sm, x))
y.gs <- as.numeric(predict(gam.gs, x))

d.pred <- data.table(x=x$freq, y.gpt2xl=y.gpt2xl, y.gpt2sm=y.gpt2sm, y.gs=y.gs)
d.pred <- melt(d.pred, id.vars="x", variable.name="source", value.name="y")
d.pred$source <- factor(d.pred$source, levels=c("y.gpt2xl", "y.gpt2sm", "y.gs"),
                        labels=c("GPT2-xl", "GPT2-sm", "human"))

p.pred.base <- ggplot(d.pred, aes(x=x, y=abs(y), color=source)) +
  geom_line(aes(color=source, linetype=source)) +
  theme_bw() +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k]))) +
  scale_color_brewer(palette="Set1") +
  scale_linetype_manual(values=c("dashed", "dotted", "solid"))

p.pred <- p.pred.base + ggtitle("Smoothed spectra")
p.pred.log <- p.pred.base + scale_y_log10() + ggtitle("Smoothed spectra (log scale)")

p.pred.combine <- ((p.pred / p.pred.log) | guide_area()) + plot_layout(ncol=2, guides = "collect", widths = c(2.5,1))
ggsave("smoothed_spectra.pdf", plot=p.pred.combine, width=5, height=4)