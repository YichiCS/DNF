import torch

def one_inversion_steps(x, seq, model, parameters):

    with torch.no_grad():
        n = x.size(0)
        t = (torch.ones(n) * seq[0]).to(x.device)
        et = model(x, t)

    return et

def inversion_steps(x, seq, model, parameters):

    with torch.no_grad():

        seq_next = seq[1:]
        seq  = seq[:-1]

        x = x.to('cuda')
        n = x.size(0)
        
        noise_bar = torch.zeros(x.shape).to(x.device)
        x0_preds = []
        noises = []
        xs = [x]
        count = 0
        

        for i, j in zip(seq, seq_next):

            t = (torch.ones(n) * i).to(x.device)
            t_next = (torch.ones(n) * j).to(x.device)

            at = parameters["alpha_cumprod"][t.long()].reshape(x.shape[0], 1, 1, 1)
            at_next = parameters["alpha_cumprod"][t_next.long()].reshape(x.shape[0], 1, 1, 1)
            
            xt = xs[-1].to("cuda")
            et = model(xt, t)
            if count == 0:
                f = et
            count = count + 1
            noise_bar = noise_bar + et
            
            x0_pred = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_pred.to('cpu'))

            xt_next = at_next.sqrt() * x0_pred + (1 - at_next).sqrt() * et
            noises.append(et.to('cpu'))
            xs.append(xt_next.to('cpu'))

        noise_bar = noise_bar
    return noise_bar / len(seq), et, f, noises