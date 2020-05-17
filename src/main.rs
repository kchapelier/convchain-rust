extern crate rand;

//use std::time::{SystemTime};

struct Field {
    width: usize,
    height: usize,
    data: Vec<bool>
}

struct WeightsCache {
    n: u8,
    weights: Vec<f32>
}

struct ConvChain<'a> {
    sample_width: u8,
    sample_height: u8,
    sample: Vec<bool>,

    cache: WeightsCache,

    rng: &'a dyn Fn() -> f32
}

fn process_weights(sample: &Vec<bool>, sample_width: u8, sample_height: u8, n: u8) -> Vec<f32> {
    let mut weights: Vec<f32> = vec![0.0; 1 << (n * n)];
    let nsize = usize::from(n);

    let pattern = |func: &dyn Fn(usize, usize) -> bool| {
        let mut result: Vec<bool> = vec![false; nsize * nsize];

        for x in 0..nsize {
            for y in 0..nsize {
                result[x + y * nsize] = func(x, y);
            }
        }

        result
    };

    let rotate = |p: &Vec<bool>| {
        let closure = |x: usize, y: usize| -> bool {
            p[nsize - 1 - y + x * nsize]
        };

        pattern(&closure)
    };

    let reflect = |p: &Vec<bool>| {
        let closure = |x: usize, y: usize| -> bool {
            p[nsize - 1 - x + y * nsize]
        };

        pattern(&closure)
    };

    let index = |p: &Vec<bool>| -> usize {
        let mut result: usize = 0;
        let mut power: usize = 1;
        let len: usize = p.len();

        for i in 0..len {
            if p[len - 1 - i] {
                result = result + power;
            }
            power = power * 2;
        }

        result
    };

    let usample_height: usize = usize::from(sample_height);
    let usample_width: usize = usize::from(sample_width);

    for y in 0..usample_height {
        for x in 0..usample_width {
            let closure = |dx, dy| -> bool {
                sample[
                    ((x + dx) % usample_width) +
                    ((y + dy) % usample_height) * usample_width
                ]
            };
            let p0: Vec<bool> = pattern(&closure);
            let p1 = rotate(&p0);
            let p2 = rotate(&p1);
            let p3 = rotate(&p2);
            let p4 = reflect(&p0);
            let p5 = reflect(&p1);
            let p6 = reflect(&p2);
            let p7 = reflect(&p3);

            weights[index(&p0)] += 1.0;
            weights[index(&p1)] += 1.0;
            weights[index(&p2)] += 1.0;
            weights[index(&p3)] += 1.0;
            weights[index(&p4)] += 1.0;
            weights[index(&p5)] += 1.0;
            weights[index(&p6)] += 1.0;
            weights[index(&p7)] += 1.0;
        }
    }

    for k in 0..weights.len() {
        if weights[k] <= 0.0 {
            weights[k] = 0.1;
        }
    }

    weights
}

fn generate_base_field(result_width: usize, result_height: usize, rng: &dyn Fn() -> f32) -> Field {
    let data_size = result_width * result_height;
    let mut data: Vec<bool> = Vec::with_capacity(data_size);

    for _i in 0..data_size {
        data.push(rng() > 0.5);
    }

    Field {
        width: result_width,
        height: result_height,
        data: data
    }
}

fn apply_changes (field: &mut Field, weights: &Vec<f32>, n: u8, temperature: f32, changes: u32, rng: &dyn Fn() -> f32) {
    let result_width: isize = field.width as isize;
    let result_height: isize = field.height as isize;
    let un: isize = isize::from(n);

    for _i in 0..changes {
        let mut q: f32 = 1.0;

        let r: isize = (rng() * ((result_width * result_height) as f32)) as isize;
        let x: isize = r % result_width;
        let y: isize = r / result_width;

        for sy in (y - un + 1)..(y + un) {
            for sx in (x - un + 1)..(x + un) {
                let mut ind: isize = 0;
                let mut difference: isize = 0;

                for dy in 0..un {
                    for dx in 0..un {
                        let power = 1 << (dy * un + dx);
                        let mut nx: isize = sx + dx;
                        let mut ny: isize = sy + dy;

                        //TODO use rem_euclidean instead ?
                        if nx < 0 {
                            nx += result_width;
                        } else if nx >= result_width {
                            nx -= result_width;
                        }

                        //TODO use rem_euclidean instead ?
                        if ny < 0 {
                            ny += result_height;
                        } else if ny >= result_height {
                            ny -= result_height;
                        }

                        let value: bool = field.data[(nx + ny * result_width) as usize];

                        if value {
                            ind = ind + power;
                        }

                        if nx == x && ny == y {
                            if value {
                                difference = power as isize;
                            } else {
                                difference = (power as isize) * -1;
                            }
                        }
                    }
                }

                q *= weights[(ind - difference) as usize] / weights[ind as usize];
            }
        }

        let data_index: usize = (x + y * result_width) as usize;

        if q >= 1.0 {
            field.data[data_index] = !field.data[data_index];
        } else {
            if temperature != 1.0 {
                q = q.powf(1.0 / temperature);
            }

            if q > rng() {
                field.data[data_index] = !field.data[data_index];
            }
        }
    }

}

impl<'a> ConvChain<'a> {
    fn new(sample_width: u8, sample_height: u8, sample: Vec<bool>) -> Self {
        let mut cc = ConvChain {
            sample_width: 0,
            sample_height: 0,
            sample: Vec::with_capacity(1),
            cache: WeightsCache {
                n: 0,
                weights: Vec::with_capacity(1)
            },
            rng: &rand::random::<f32>
        };

        cc.set_sample(sample_width, sample_height, sample);

        cc
    }

    fn set_sample(&mut self, sample_width: u8, sample_height: u8, sample: Vec<bool>) {
        if sample.len() != usize::from(sample_width) * usize::from(sample_height) {
            panic!("crash and burn");
        }
        
        self.sample_width = sample_width;
        self.sample_height = sample_height;
        self.sample = sample;

        // invalidate cached weights
        self.cache.n = 0;
    }

    fn get_weights(&mut self, n: u8) -> &Vec<f32> {
        if self.cache.n != n {
            self.cache.n = n;
            self.cache.weights = process_weights(&self.sample, self.sample_width, self.sample_height, n);
        }

        &self.cache.weights
    }

    fn set_rng(&mut self, rng: &'a dyn Fn() -> f32) {
        self.rng = rng;
    }

    fn initialize_field(&mut self, width: usize, height: usize) -> Field {
        generate_base_field(width, height, self.rng)
    }

    fn iterate(&mut self, field: &mut Field, n: u8, temperature: f32, tries: u32) {
        let rng = self.rng;
        let weights = self.get_weights(n);

        apply_changes(field, weights, n, temperature, tries, rng);
    }
}

fn main() {
    /*
    let sizes = [64, 256, 1024, 2048];
    let mut s = true;

    for i in 0..sizes.len() {
        let v: Vec<bool> = vec![
             true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
             true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
             true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
             true,  true,  true, false, false, false, false,  true,  true,  true,
            false, false, false, false, false, false, false, false, false, false,
            false, false, false, false, false, false, false, false, false, false,
             true,  true,  true, false, false, false, false,  true,  true,  true,
             true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
             true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
             true,  true,  true,  true,  true,  true,  true,  true,  true,  true
        ];

        let time = SystemTime::now();
        let mut cc = ConvChain::new(10, 10, v);
        cc.set_rng(&rand::random::<f32>);

        let mut field = cc.initialize_field(sizes[i], sizes[i]);

        cc.iterate(&mut field, 3, 0.5, 2000);

        println!("10x10 sample / {0}x{0} field / 2000 iterations => {1:?}ms", sizes[i], time.elapsed());

        s = s ^ field.data[1];
    }

    println!("{}", s);
    */

    let v: Vec<bool> = vec![
        true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
        true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
        true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
        true,  true,  true, false, false, false, false,  true,  true,  true,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        true,  true,  true, false, false, false, false,  true,  true,  true,
        true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
        true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
        true,  true,  true,  true,  true,  true,  true,  true,  true,  true
    ];

    let mut cc = ConvChain::new(10, 10, v);
    cc.set_rng(&rand::random::<f32>);

    let mut field = cc.initialize_field(48, 16);

    cc.iterate(&mut field, 3, 0.5, 3000);

    for y in 0..field.height {
        for x in 0..field.width {
            print!("{}", if field.data[x + y * field.width] { "█" } else { " " });
        }
        println!();
    }
}
