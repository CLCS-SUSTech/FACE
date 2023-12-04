require("ggplot2")
require("data.table")

d.human <- fread("../data/demo_human.fft.txt")
d.model <- fread("../data/demo_model.fft.txt")

# Add sequence ID
add_sid <- function(dt) {
  dt[, freq2 := shift(freq, 1, type="lead", fill=0.5)]
  dt$diffSeries <- dt$freq > dt$freq2
  dt$sid <- cumsum(dt$diffSeries)
  dt$sid <- shift(dt$sid, 1, type="lag", fill=0)
  dt
}
d.human <- add_sid(d.human)
d.model <- add_sid(d.model)

# Add type column
d.human$type <- "human"
d.model$type <- "model"
d.combined <- rbind(d.human, d.model)

# Smoothed plot
p <- ggplot(d.combined, aes(freq, power, color=type)) +
  geom_smooth() +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, vjust=-8, size = 12)) +
  ggtitle("Human vs. Model") +
  labs(x = bquote(omega[k]), y = bquote(X(omega[k])))
ggsave("demo_spectrum_smoothed.pdf", plot=p, width=5, height=5)