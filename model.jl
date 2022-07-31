using Pkg
using CSV
using DataFrames
using Flux, Images, Plots
using Random, Statistics

Random.seed!(1)
sinais =  {"0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "5": "F", "6": "G", "7": "H", "8": "I", 
           "10": "K", "11": "L", "12": "M", "13": "N", "14": "O", "15": "P", "16": "Q", "17": "R",
        "18": "S", "19": "T", "20": "U", "21": "V", "22": "W", "23": "X", "24": "Y" }

X_train_raw, y_train_raw  = CSV.File("/Users/mariochiaparini/Desktop/kaggles_machine_learning/data/sign_mnist_train.csv")
X_test_raw, y_test_raw = CSV.File("/Users/mariochiaparini/Desktop/kaggles_machine_learning/data/sign_mnist_test.csv")

X_train_raw

index = 1

img = X_train_raw[:, :, index]

colorview(Gray, img)

y_train_raw

y_train_raw[index]

X_test_raw

img = X_test_raw[:, :, index]

colorview(Gray, img)

y_test_raw

y_test_raw[index]

X_train = Flux.flatten(X_train_raw)

X_test = Flux.flatten(X_test_raw)

y_train = onehotbatch(y_train_raw, 0:9)

y_test = onehotbatch(y_test_raw, 0:9)

model = Chain(
    Dense(28 * 28, 32, relu),
    Dense(32, 10),
    softmax
)

loss(x, y) = crossentropy(model(x), y)

ps = params(model)

learning_rate = Float32(0.01)

opt = ADAM(learning_rate)

loss_history = []

epochs = 500

for epoch in 1:epochs
    train!(loss, ps, [(X_train, y_train)], opt)
    train_loss = loss(X_train, y_train)
    push!(loss_history, train_loss)
    println("Epoch = $epoch : Training Loss = $train_loss")
end

y_hat_raw = model(X_test)

y_hat = onecold(y_hat_raw) .- 1
y = y_test_raw
mean(y_hat .== y)

check = [y_hat[i] == y[i] for i in 1:length(y)]

index = collect(1:length(y))

check_display = [index y_hat y check]

vscodedisplay(check_display)



misclass_index = 9

img = X_test_raw[:, :, misclass_index]

colorview(Gray, img")

y[misclass_index]

y_hat[misclass_index]
gr(size = (600, 600))

p_l_curve = plot(1:epochs, loss_history, 
    xlabel = "Epochs",
    ylabel = "Loss",
    title = "Learning Curve",
    legend = false,
    color = :blue,
    linewidth = 2
)


savefig(p_l_curve, "/Users/mariochiaparini/Desktop/kaggles_machine_learning/computervision/libras_julia.jl")